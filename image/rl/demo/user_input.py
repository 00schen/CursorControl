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
from tqdm import tqdm

from stable_baselines3.sac import SAC
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from envs import *
dirname = os.path.dirname(os.path.abspath(__file__))

class FollowerAgent:
	def reset(self):
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
		return [self.trajectory*1.0]

class BadTargetAgent:
	def __init__(self,env):
		self.target = self.generate_target()
		self.env = env
	def predict(self):
		self.target = self.target if rng.random() < .8 else self.generate_target()
		return [self.target-self.env.tool_pos]
	def generate_target(self):
		return np.array([0,-.25,1])+np.array([1,.75,.5])*rng.uniform(-1,1,3)

if __name__ == "__main__":
	curriculum_default_config = {
		'end_early': True,
		'noise': False,
		'oracle_size': 6,
		'phi':lambda d:0,
		'oracle': 'user',
		"threshold": 0.562928957579239,
		'blank': 1,
		"input_penalty": 2,
		# 'env_kwargs': {'num_targets': -1},
		'num_obs': 7,
		'num_nonnoop': 7,
		'traj_clip': .1,
		'step_limit': int(500)
	}
	env_config = deepcopy(default_config)
	env_config.update(curriculum_default_config)
	env_config.update(env_map["Laptop"])
	env_config.update(action_map['trajectory'])
	config = env_config

	env = make_vec_env(lambda: PrevNnonNoopK(env_config))
	seed = int(rng.integers(low=1000,high=int(1e10)))
	env.seed(seed)
	f_agent = FollowerAgent()
	b_agent = BadTargetAgent(env.envs[0].env.env)

	distance = []
	diff_distance = []
	cos = []
	length = []
	recommends = []
	observations = []
	dones = []

	env.render()
	info = {'recommend': np.zeros(6)}
	obs = env.reset()[0]
	f_agent.reset()

	done_count = 0
	follow_p = rng.uniform(.2,.8)
	while done_count < 10:
		print(done_count)
		
		"""action"""
		if norm(env.envs[0].env.env.target_pos-env.envs[0].env.env.tool_pos) < .15:
			actions = [[env.envs[0].env.env.target_pos-env.envs[0].env.env.tool_pos + np.array([0,0,.05])],
						np.array(f_agent.predict(info['recommend'])),
						np.array(b_agent.predict())*.05]
			action = rng.choice(actions,p=[.3,.6,.1])
		else:
			actions = [[env.envs[0].env.env.target_pos-env.envs[0].env.env.tool_pos],
					f_agent.predict(info['recommend']),
					b_agent.predict()]
			action = rng.choice(actions,p=[0,follow_p,1-follow_p])


		obs,r,done,info = env.step(action)
		obs,r,done,info = obs[0],r[0],done[0],info[0]

		if True:
			distance.append(info['distance_to_target'])
			diff_distance.append(info['diff_distance'])
			cos.append(info['cos_error'])
			length.append(norm(info['trajectory']))
			recommends.append(info['recommend'])
			observations.append(info['current_obs'])
			dones.append(done)
		
		if done:
			f_agent.reset()
			done_count += 1
			follow_p = rng.uniform(.2,.8)
	for i in range(100):
		print(len(distance))
	np.savez_compressed(f"L.user_data.3.{seed}",a=distance,b=diff_distance,c=cos,d=length,e=recommends,f=observations,g=done)

		