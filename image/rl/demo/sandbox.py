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

if __name__ == "__main__":
	curriculum_default_config = \
	{'oracle': 'trajectory', 'oracle_size': 3, 'step_limit': 200, 'env_kwargs':{'target_first': False}}
	env_config = deepcopy(default_config)
	env_config.update(curriculum_default_config)
	env_config.update(env_map["Reach"])
	env_config.update(action_map['joint'])
	config = env_config

	env = make_vec_env(lambda: TargetRegion(config))
	env = VecNormalize(env,norm_reward=False)
	agent = IKAgent(env.envs[0].env.env)

	env.render()
	obs = env.reset()[0]
	while True:
		action = [agent.predict(obs)] if np.random.random() > -1 else [env.action_space.sample()]
		# obs,r,done,info = env.step([env.envs[0].env.env.target_pos-env.envs[0].env.env.tool_pos]) 
		obs,r,done,info = env.step(action)
		obs,r,done,info = obs[0],r[0],done[0],info[0]
		print(info['distance_target'])
		# print("target_pos", env.envs[0].env.env.target_pos)
		env.render()

