import os
import argparse
from collections import deque
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
from copy import deepcopy

import assistive_gym
import gym
from gym import spaces

import numpy as np
import ray
from tqdm import tqdm
import torch

from stable_baselines3.sac import MlpPolicy
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import BaseCallback,CallbackList,CheckpointCallback
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize,SubprocVecEnv

from utils import Noise,dirname,Supervised,default_config,env_map

parser = argparse.ArgumentParser()
parser.add_argument('--config',type=int,help='dir to save trials')
parser.add_argument('--local_dir',help='dir to save trials')
args, _ = parser.parse_known_args()

class Baseline1(assistive_gym.FeedingJacoEnv):
	def __init__(self,config,time_limit=200):
		super().__init__()
		self.success_count = deque([0]*20,20)
		self.time_limit = time_limit

	def step(self,action):
		obs,r,done,info = super().step(action)
		self.timesteps += 1
		if self.timesteps >= self.time_limit:
			done = True
			self.success_count.append(self.task_success > 0)
		return obs,r,done,info
	
	def reset(self):
		obs = super().reset()
		self.timesteps = 0
		return obs

class Baseline2(Baseline1):
	def __init__(self,config,time_limit=200):
		super().__init__(config)
		self.noise = Noise(spaces.Box(-1*np.ones(3),np.ones(3)),3)

	def step(self,action):
		obs,r,done,info = super().step(action)
		recommendation = self.noise(obs[7:10])
		obs = np.concatenate((obs[:7],recommendation,obs[10:]))
		return obs,r,done,info
	
	def reset(self):
		self.noise.reset()
		obs = super().reset()
		recommendation = self.noise(obs[7:10])
		obs = np.concatenate((obs[:7],recommendation,obs[10:]))
		return obs

class Baseline3(Baseline1):
	def __init__(self,config,time_limit=200):
		super().__init__(config)
		baseline_path = os.path.join(dirname,'baseline','baseline.h5')
		config['save_path'] = baseline_path
		self.decoder = Supervised(config)
		X,Y,_Z,W = np.load(os.path.join(dirname,'F.noised_trajectory.npz')).values()
		# X = np.concatenate((X[...,:7],X[...,10:],Y),axis=2)
		X = np.concatenate((X[...,:7],X[...,10:],X[...,7:10]),axis=2)
		X = X[:1000]
		self.decoder.pretrain(X)
		self.noise = Noise(spaces.Box(-1*np.ones(3),np.ones(3)),3)
		self.observation_space = spaces.Box(-1*np.ones(28),np.ones(28))

	def step(self,action):
		obs,r,done,info = super().step(action)
		recommendation = self.noise(obs[7:10])
		obs = np.concatenate((obs[:7],obs[10:],recommendation))
		obs = self.decoder.predict(obs)
		return obs,r,done,info

	def reset(self):
		self.decoder.reset()
		self.noise.reset()
		obs = super().reset()
		recommendation = self.noise(obs[7:10])
		obs = np.concatenate((obs[:7],obs[10:],recommendation))
		obs = self.decoder.predict(obs)
		return obs

class TensorboardCallback(BaseCallback):
	def __init__(self, verbose=0):
		super().__init__(verbose)

	def _on_step(self) -> bool:
		env = self.training_env.envs[0]
		self.logger.record('success rate', np.mean(env.success_count))
		return True
class ExtendedCheckpointCallback(BaseCallback):
	def __init__(self, save_freq, save_path, verbose=0):
		super().__init__(verbose)
		self.save_freq = save_freq
		self.save_path = save_path

	def _init_callback(self):
		# Create folder if needed
		if self.save_path is not None:
			os.makedirs(self.save_path, exist_ok=True)

	def _on_step(self):
		if self.n_calls % self.save_freq == 0:
			os.makedirs(os.path.join(self.save_path, f'decoder_{self.num_timesteps}_steps'),exist_ok=True)
			decoder_path = os.path.join(self.save_path, f'decoder_{self.num_timesteps}_steps','decoder.h5')
			norm_path = os.path.join(self.save_path, f'norm_{self.num_timesteps}_steps')
			self.training_env.envs[0].decoder.model.save_weights(decoder_path,save_format='h5')
			torch.save({
				'decoder_norm':self.training_env.envs[0].decoder.norm,
			},norm_path)
		return True
class PredictorTrainCallback(BaseCallback):
	def _on_step(self):
		return True
	def _on_rollout_end(self):
		self.training_env.envs[0].decoder.train()

# @ray.remote(num_cpus=1,num_gpus=1)
class BaselineRunner:
	def run(self,config):
		exp_name,baseline,callbacks = config['exp_name'],config['baseline'],config['callbacks']
		logdir = os.path.join(args.local_dir,exp_name)
		make_callbacks = {
			'PredictorTrain': lambda: PredictorTrainCallback(),
			'Tensorboard': lambda: TensorboardCallback(),
			'ExtendedCheckpoint': lambda: ExtendedCheckpointCallback(save_freq=int(1e4),save_path=logdir),
			'Checkpoint': lambda: CheckpointCallback(save_freq=int(1e4), save_path=logdir),
		}
		
		eval_env = baseline(config)
		env = make_vec_env(lambda: baseline(config),monitor_dir=logdir)
		env = VecNormalize(env,norm_reward=False)

		tensorboard_path = os.path.join(args.local_dir,'tensorboard')
		os.makedirs(tensorboard_path, exist_ok=True)
		model = SAC(MlpPolicy, env, learning_rate=1e-5, verbose=1,tensorboard_log=tensorboard_path)

		callback = CallbackList([make_callbacks[callback]() for callback in callbacks])
		time_steps = int(1e6)

		model.learn(total_timesteps=time_steps,eval_env=eval_env,callback=callback,tb_log_name=exp_name)
		env.save_running_average(logdir)


if __name__ == "__main__":
	# ray.init(temp_dir='/tmp/ray_exp')

	# runners = [BaselineRunner.remote() for i in range(3)]
	# configs = [
	# 	{'exp_name': '6_8_4', 'baseline': Baseline3, 'callbacks': ['PredictorTrain','Tensorboard','ExtendedCheckpoint','Checkpoint'],},
	# 	{'exp_name': '6_8_3', 'baseline': Baseline2, 'callbacks': ['Tensorboard','Checkpoint'],},
	# 	{'exp_name': '6_8_2', 'baseline': Baseline1, 'callbacks': ['Tensorboard','Checkpoint'],},
	# ]

	# runs = [runner.run.remote(config) for runner,config in zip(runners,configs)]
	# runs = [ray.get(run) for run in runs]

	configs = [
		{'exp_name': '6_9_0', 'baseline': Baseline1, 'callbacks': ['Tensorboard','Checkpoint'],},
		{'exp_name': '6_9_1', 'baseline': Baseline2, 'callbacks': ['Tensorboard','Checkpoint'],},
		{'exp_name': '6_9_2', 'baseline': Baseline3, 'callbacks': ['PredictorTrain','Tensorboard','ExtendedCheckpoint','Checkpoint'],},
	]

	config = deepcopy(default_config)
	config.update(env_map['FeedingJaco-v1'])
	config.update(configs[args.config])
	BaselineRunner().run(config)
