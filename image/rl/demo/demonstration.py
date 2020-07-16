import gym
import time
import os,sys
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf
import torch
import numpy as np
from copy import deepcopy
from types import MethodType
import pybullet as p
from types import SimpleNamespace

from stable_baselines3.sac import SAC
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecVideoRecorder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from envs import *
dirname = os.path.dirname(os.path.abspath(__file__))

class DemonstrationEnv(SharedAutonomy):
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

if __name__ == "__main__":
	env_config = deepcopy(default_config)
	env_config.update(env_map["Laptop"])
	env_config.update({
		'end_early': True,
		'noise': False,
		'oracle_size': 6,
		'phi':lambda d: 0,
		'oracle': 'ded_target',
		'blank': 1,
		'oracle': 'user_model',
		"input_penalty": 2,
		"target_max": 20,
		'action_type': 'cat_target'
	})

	env = DemonstrationEnv(env_config)
	base_env = env.env

	obs_data = []
	action_data = []
	r_data = []
	done_data = []
	index_data = []
	target_data = []
	
	while env.target_count < env.target_max:
		obs = env.reset()
		f_probs = np.zeros(base_env.num_targets)
		f_probs[base_env.target_index] = 2
		agent = SimpleNamespace(predict=lambda obs: rng.choice(base_env.num_targets,p=softmax(f_probs)))
		done = False
		
		obs_ep = []
		action_ep = []
		r_ep = []
		done_ep = []
		target_ep = []
		print(base_env.target_index, env.target_count)

		while not done:
			action = agent.predict(obs)
			# print(base_env.target_pos)
			# print(action)

			obs_ep.append(obs)
			action_ep.append(action)

			obs,r,done,info = env.step(action)

			r_ep.append(r)
			done_ep.append(done)
			target_ep.append(env.env.targets)
		obs_ep.append(obs)

		# if base_env.task_success > 0:
		obs_data.append(obs_ep)
		action_data.append(action_ep)
		r_data.append(r_ep)
		done_data.append(done_ep)
		index_data.append(base_env.target_index)
		target_data.append(target_ep)
		env.new_target()

	np.savez_compressed(f"{env_config['env_name'][:2]}_demo1", obs=obs_data,action=action_data,r=r_data,done=done_data,index=index_data,target=target_data)
	env.close()


