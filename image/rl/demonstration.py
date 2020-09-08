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


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import *
dirname = os.path.dirname(os.path.abspath(__file__))

from discrete_experiment import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int,help='dir to save trials')
parser.add_argument('--env_name',help='dir to save trials')
args, _ = parser.parse_known_args()


# import ray

if __name__ == "__main__":
	env_name = args.env_name
	wrapper = lambda env_name: reduce(lambda value,func: func(value),
				[sparse_factory,shared_autonomy_factory,demonstration_factory],env_name)

	@ray.remote(num_cpus=1,num_gpus=0)
	class Sampler:
		def sample(self,seed,count):
			env_config = deepcopy(default_config)
			env_config.update(env_map[env_name])
			env_config.update(dict(
				oracle_size=6,
				oracle='user_model',
				num_obs=5,
				num_nonnoop=5,
				threshold=.3,
				input_penalty=.01,
				action_type='trajectory',
				action_penalty=0,
				include_target=False,
				cap=0,
				step_limit=200,
				target_max=10,
				action_clip=.05,
			))

			env = feedback_factory(wrapper(env_config['env_name']))(env_config)
			env.seed(seed)
			base_env = env.env
			env.render('human')

			paths = []
			fail_paths = deque([],1)
			success_paths = deque([],1)
			while env.target_count < env.target_max:
				obs = env.reset()
				# policy.reset()

				user_input = np.zeros(3)
				path = {'observations':[],'actions':[],'next_observations':[],'rewards':[],'terminals':[],'agent_infos':[],'env_infos':[]}
				print(base_env.target_index, env.target_count)

				# if rng.random() > .5:
				# p_0 = rng.random()*.6+.2
				p_0 = 1
				agents = [FollowerAgent(),EpsilonAgent(epsilon=1/10)]
				p=[p_0,1-p_0]
				agent = SimpleNamespace(predict=lambda info: rng.choice([agent.predict(info) for agent in agents],p=p))
				# else:
				# 	agents = [FollowerAgent(),EpsilonAgent(epsilon=1/10),ClosestAgent()]
				# 	p=[.2,.8]
				# 	agent = SimpleNamespace(predict=lambda info: rng.choice([agent.predict(info) for agent in agents],p=p))

				done = False
				info = {'recommend': np.zeros(6), 'targets': base_env.targets, 'target_index': base_env.target_index, 'current_observation': obs}
				while not done:
					action,info['action_index'] = agent.predict(info)
					# action = agent.predict(obs)

					path['observations'].append(obs)
					path['actions'].append(action)
					
					# action_index = np.zeros(6)
					# action_index[info['action_index']] = 1
					# obs,r,done,info = env.step(action_index)
					print(info['recommend'],action)
					obs,r,done,info = env.step(action)
					info['current_observation'] = obs

					path['next_observations'].append(obs)
					path['rewards'].append(r)
					path['terminals'].append(done)
					path['agent_infos'].append({})
					path['env_infos'].append(info)
			
				if base_env.task_success:
					success_paths.append(path)
				else:
					fail_paths.append(path)
				if len(success_paths) == 1 and len(fail_paths) == 1:
					# paths.append(path)
					paths.extend(success_paths)
					paths.extend(fail_paths)
					env.new_target()
					fail_paths = deque([],1)
					success_paths = deque([],1)

			return paths

	# ray.init(temp_dir='/tmp/ray_exp')
	# num_workers = 10
	# count = 120//num_workers

	# samplers = [Sampler.remote() for i in range(num_workers)]
	# samples = [samplers[i].sample.remote(args.seed+i,count) for i in range(num_workers)]
	# samples = [ray.get(sample) for sample in samples]

	# paths = list(sum(samples,[]))

	# np.save(os.path.join("demos",f"{env_name}_user_model_{args.seed}"), paths)

	paths = Sampler().sample(args.seed,100)