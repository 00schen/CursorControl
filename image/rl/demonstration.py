import gym
import time
import os,sys
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import numpy as np
from copy import deepcopy
from types import MethodType
import pybullet as p
from types import SimpleNamespace

import ray

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import *
dirname = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int,help='dir to save trials')
parser.add_argument('--env_name',help='dir to save trials')
args, _ = parser.parse_known_args()

def demonstration_factory(base):
	class DemonstrationEnv(base):
		def __init__(self,config):
			super().__init__(config)
			self.target_index = 0
			self.target_count = 0
			self.target_max = config['target_max']

		def new_target(self):
			self.target_index += 1
			if self.target_index == self.env.num_targets:
				self.target_index %= self.env.num_targets
				self.target_count += 1
		def reset(self):
			target_index = self.target_index
			def generate_target(self,index):
				nonlocal target_index
				self.__class__.generate_target(self,target_index)
			self.env.generate_target = MethodType(generate_target,self.env)
			return super().reset()
	return DemonstrationEnv

class FollowerAgent:
	def __init__(self):
		self.trajectory = np.array([0,0,0])
		self.action_count = 0
	def predict(self,info):
		recommend = info['recommend']
		if np.count_nonzero(recommend):
			index = np.argmax(recommend)
			self.trajectory = {
				0: np.array([-1,0,0]),
				1: np.array([1,0,0]),
				2: np.array([0,-1,0]),
				3: np.array([0,1,0]),
				4: np.array([0,0,-1]),
				5: np.array([0,0,1]),
			}[index]
			# self.action_count = 0
		# self.action_count += 1
		# if self.action_count >= 10:
		# 	self.trajectory = np.array([0,0,0])
		tool_pos = info['current_observation'][:3]
		target_index = min(range(len(info['targets'])),key=lambda i: norm(tool_pos+self.trajectory-info['targets'][i]))
		return self.trajectory*1.0, target_index

class EpsilonAgent:
	def __init__(self,epsilon=.25):
		self.epsilon = epsilon
		self.action_index = 0
	def predict(self,info):
		target = info['targets'][info['target_index']]
		tool_pos = info['current_observation'][:3]
		real_traj = target - tool_pos

		real_index = np.argmax(np.abs(real_traj))
		real_index = real_index*2 + real_traj[real_index] > 0

		if self.action_index == real_index:
			self.action_index = rng.choice(6)
		else:
			self.action_index = self.action_index if rng.random() > self.epsilon else rng.choice(6)

		trajectory = traj = [
					np.array((-1,0,0)),
					np.array((1,0,0)),
					np.array((0,-1,0)),
					np.array((0,1,0)),
					np.array((0,0,-1)),
					np.array((0,0,1)),
				][self.action_index]
		
		return trajectory, self.action_index

if __name__ == "__main__":
	env_name = args.env_name
	wrapper = lambda env_name: reduce(lambda value,func: func(value),
				[sparse_factory,shared_autonomy_factory,demonstration_factory],env_name)

	@ray.remote(num_cpus=1,num_gpus=0)
	class Sampler:
		def sample(self,seed,count):
			env_config = deepcopy(default_config)
			env_config.update(env_map[env_name])
			env_config.update({
				'oracle_size': 6,
				'oracle': 'user_model',
				'num_nonnoop': 10,
				'num_obs': 10,
				"input_penalty": 2,
				'action_type': 'trajectory',
				# 'action_type': 'target',
				'threshold': .15,
				'action_penalty': 0,
				# 'env_kwargs': {'num_targets': 4},
				"target_max": count,
				'step_limit': 200,
			})

			env = wrapper(env_config['env_name'])(env_config)
			env.seed(seed)
			base_env = env.env
			# env.render('human')

			paths = []
			round_paths = deque([],5)
			task_success = deque([],5)
			while env.target_count < env.target_max:
				obs = env.reset()
				user_input = np.zeros(3)
				path = {'observations':[],'actions':[],'next_observations':[],'rewards':[],'terminals':[],'agent_infos':[],'env_infos':[]}
				print(base_env.target_index, env.target_count)

				# if rng.random() > .5:
				agents = [FollowerAgent(),EpsilonAgent(epsilon=1/10)]
				p=[.8,.2]
				agent = SimpleNamespace(predict=lambda info: rng.choice([agent.predict(info) for agent in agents],p=p))
				# else:
				# 	agents = [FollowerAgent(),EpsilonAgent(epsilon=1/10),ClosestAgent()]
				# 	p=[.2,.8]
				# 	agent = SimpleNamespace(predict=lambda info: rng.choice([agent.predict(info) for agent in agents],p=p))

				done = False
				info = {'recommend': np.zeros(6), 'targets': base_env.targets, 'target_index': base_env.target_index, 'current_observation': obs}
				while not done:
					action,info['action_index'] = agent.predict(info)

					path['observations'].append(obs)
					path['actions'].append(action)

					obs,r,done,info = env.step(action)
					info['current_observation'] = obs

					path['next_observations'].append(obs)
					path['rewards'].append(r)
					path['terminals'].append(done)
					path['agent_infos'].append({})
					path['env_infos'].append(info)
			
				round_paths.append(path)
				task_success.append(base_env.task_success)
				if np.mean(task_success) > 0 and len(round_paths) > 4:
					# paths.append(path)
					paths.extend(round_paths)
					env.new_target()
					round_paths = deque([],5)
					task_success = deque([],5)

			return paths

	ray.init(temp_dir='/tmp/ray_exp')
	num_workers = 10
	count = 100//num_workers

	samplers = [Sampler.remote() for i in range(num_workers)]
	samples = [samplers[i].sample.remote(args.seed+i,count) for i in range(num_workers)]
	samples = [ray.get(sample) for sample in samples]

	paths = list(sum(samples,[]))

	np.save(os.path.join("demos",f"{env_name}_user_model_{args.seed}"), paths)

	# paths = Sampler().sample(args.seed,100)