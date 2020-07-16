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

from stable_baselines3.sac import SAC
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecVideoRecorder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from envs import *
dirname = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
	env_config = deepcopy(default_config)
	env_config.update(env_map["LightSwitch"])
	env_config.update(action_map['disc_target'])
	env_config.update({
		'end_early': True,
		'noise': False,
		'oracle_size': 6,
		'phi':lambda d: 0,
		'oracle': 'dd_target',
		'num_obs': 6,
		# 'num_obs': 4,
		'num_nonnoop': 10,
		'blank': 1,
		'oracle': 'user_model',
		"input_penalty": 2,
	})

	env = make_vec_env(lambda: PrevNnonNoopK(env_config))
	env = VecNormalize.load("norm.750000",env)
	# env = VecNormalize.load("norm.1000000",env)
	# env = VecVideoRecorder(env, 'videos/',
    #                    record_video_trigger=lambda x: x == 0, video_length=1000,
    #                    name_prefix="light_switch")
	agent = SAC.load("model.750000")
	# agent = SAC.load("model.1000000")

	env.render('human')
	# p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "light_switch.mp4")

	obs = env.reset()[0]

	# while True:
	i = 0
	while i < 10:
		action = agent.predict(obs)
		obs,r,done,info = env.step(action)
		obs,r,done,info = obs[0],r[0],done[0],info[0]
		if done:
			i+=1
	env.close()


