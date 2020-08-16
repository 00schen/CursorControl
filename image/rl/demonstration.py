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

# import ray

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utils import *
from envs import *
dirname = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
# parser.add_argument('--count',type=int,help='dir to save trials')
parser.add_argument('--seed',type=int,help='dir to save trials')
parser.add_argument('--env_name',help='dir to save trials')
# parser.add_argument('--demo',type=int,help='dir to save trials')
# parser.add_argument('--dense',type=int,help='dir to save trials')
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
	def predict(self,recommend):
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
			self.action_count = 0
		self.action_count += 1
		if self.action_count >= 10:
			self.trajectory = np.array([0,0,0])
		return self.trajectory*1.0

if __name__ == "__main__":
	env_name = args.env_name
	wrapper = lambda env_name: reduce(lambda value,func: func(value),
				[sparse_factory,shared_autonomy_factory,demonstration_factory],env_name)

	# @ray.remote(num_cpus=1,num_gpus=0)
	class Sampler:
		def sample(self,seed,count):
			env_config = deepcopy(default_config)
			env_config.update(env_map[env_name])
			env_config.update({
				'oracle_size': 6,
				# 'oracle': 'ded_target' if args.dense else 'random_traj',
				'oracle': 'user',
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
			env.render('human')

			paths = []
			while env.target_count < env.target_max:
				obs = env.reset()
				user_input = np.zeros(3)
				path = {'observations':[],'actions':[],'next_observations':[],'rewards':[],'terminals':[],'agent_infos':[],'env_infos':[]}
				print(base_env.target_index, env.target_count)

				# noise = .1 if args.demo else .3
				# agent = SimpleNamespace(predict=lambda obs: base_env.target_pos-base_env.tool_pos+noise*rng.uniform(-1,1,3))
				# agent = SimpleNamespace(predict=lambda obs: user_input if rng.random() > .3 else base_env.target_pos-base_env.tool_pos)				
				# agent = SimpleNamespace(predict=lambda obs: base_env.target_pos+noise*rng.uniform(-1,1,3))
				agent = FollowerAgent()

				done = False
				info = {'recommend': np.zeros(6)}
				while not done:
					action = agent.predict(info['recommend'])

					path['observations'].append(obs)
					path['actions'].append(action)

					obs,r,done,info = env.step(action)
					# if not info['noop']:
					# 	index = np.argmax(info['recommend'])
					# 	user_input = np.zeros(3)
					# 	user_input[index//2] = -1 if not index % 2 else 1
					# 	noise = .1 if args.demo else info['distance_to_target']
					# if info['distance_to_target'] < .1:
					# 	user_input = base_env.target_pos-base_env.tool_pos

					path['next_observations'].append(obs)
					path['rewards'].append(r)
					path['terminals'].append(done)
					path['agent_infos'].append({})
					path['env_infos'].append(info)
			
				# print(f"number of success: {base_env.task_success}")
				# if base_env.task_success > 0 or not args.demo:
				if base_env.task_success > 0:
					paths.append(path)
					env.new_target()

			return paths

	# ray.init(temp_dir='/tmp/ray_exp')
	# num_workers = 10
	# count = args.count

	# samplers = [Sampler.remote() for i in range(num_workers)]
	# samples = [samplers[i].sample.remote(1000+i,count) for i in range(num_workers)]
	# samples = [ray.get(sample) for sample in samples]

	# paths = list(sum(samples,[]))

	# np.save(f"demos/{env_name}_{'demo' if args.demo else 'offpolicy'}{1 if args.dense else 3}_100", paths)
	# # np.save(f"demos/{env_name}_offpolicy{1 if args.dense else 3}b", paths)

	paths = Sampler().sample(args.seed,2)
	np.save(os.path.join("demos",f"{env_name}_user_{args.seed}"), paths)