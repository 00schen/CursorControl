import gym
import time
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf
import torch
import numpy as np
from scipy.spatial.distance import cosine
from copy import deepcopy

from stable_baselines3.sac import MlpPolicy
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import CallbackList,BaseCallback,CheckpointCallback
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from utils import deque,RunningMeanStd,spaces,Supervised,SparseEnv,default_config,env_map

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir',help='dir to save trials')
parser.add_argument('--config',type=int,help='dir to save trials')
args, _ = parser.parse_known_args()

class RewardEnv(SparseEnv):
	reward_map = {
		'trajectory': lambda self: lambda act,rec: np.linalg.norm(act-rec),
		'target': lambda self: lambda act,rec: cosine(rec,act-self.env.tool_pos)
	}

	def __init__(self,config):
		super().__init__(config)
		self.gamma = 10
		self.phi = RewardEnv.reward_map[config['oracle']](self)
		self.action_space = spaces.Box(-1*np.ones(3),np.ones(3))
		self.observation_space = spaces.Box(-1*np.ones(config['obs_size']),np.ones(config['obs_size']))

	def step(self,action):
		obs,r,done,info = super().step(action)
		dist = self.phi(action,obs[-3:])
		r += self.gamma*(self.old_dist - dist)
		self.old_dist = dist

		return obs,r,done,info

	def reset(self):
		obs = super().reset()
		self.old_dist = self.phi(self.action_space.sample(),obs[-3:])
		return obs

class TensorboardCallback(BaseCallback):
	def _on_step(self) -> bool:
		env = self.training_env.envs[0]
		self.logger.record('success rate', np.mean(env.success_count))
		return True

class RewardRunner:
	def run(self,config):
		logdir = os.path.join(args.local_dir,config['exp_name'])

		eval_env = RewardEnv(config)
		env = make_vec_env(lambda: RewardEnv(config),monitor_dir=logdir)
		env = VecNormalize(env,norm_reward=False)
		tensorboard_path = os.path.join(args.local_dir,'tensorboard')
		os.makedirs(tensorboard_path, exist_ok=True)
		model = SAC(MlpPolicy, env, verbose=1,tensorboard_log=tensorboard_path)

		callback = CallbackList([
				TensorboardCallback(),
				CheckpointCallback(save_freq=int(1e4), save_path=logdir),
				])
		time_steps = int(1e6)
		model.learn(total_timesteps=time_steps,callback=callback,tb_log_name=config['exp_name'])

if __name__ == "__main__":
	configs = [
		{'exp_name': '6_9_3', 'env1': 'FeedingJaco-v1', 'oracle': 'trajectory'},
		{'exp_name': '6_9_4', 'env1': 'FeedingJaco-v1', 'oracle': 'target'},
		{'exp_name': '6_9_5', 'env1': 'ScratchItchJaco-v1', 'oracle': 'trajectory'},
		{'exp_name': '6_9_6', 'env1': 'ScratchItchJaco-v1', 'oracle': 'target'},
	]

	config = deepcopy(default_config)
	config.update(configs[args.config])
	config.update(env_map[config['env1']])
	RewardRunner().run(config)
