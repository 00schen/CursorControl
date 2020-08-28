import numpy as np

import torch
from tensorflow import keras
from tensorflow.keras.layers import Dense

import pybullet as p

import os
import pickle

from envs import *
from collections import namedtuple
from tqdm import tqdm,trange

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
import torch.nn.functional as F
class PathAdaptLoader(DictToMDPPathLoader):
	def load_path(self, path, replay_buffer, obs_dict=None):
		replay_buffer.env.adapt_path(path)
		super().load_path(path, replay_buffer, obs_dict)

def checkoff_adapt(self,path):
	done_iter = iter(path['terminals'])
	info_iter = iter(path['env_infos'])

	done = False
	while not done:
		info = next(info_iter)
		info['adapt'] = False
		info['offline'] = True
		done = next(done_iter)
	return path

def switch_adapt(self,path):
	obs_iter = iter(path['observations']+[path['next_observations'][-1]])
	done_iter = iter(path['terminals'])
	new_path = {'observations':[],'next_observations':[]}

	obs = next(obs_iter)
	obs = np.concatenate((obs[:-10],obs[-7:]))
	done = False
	while not done:
		new_path['observations'].append(obs)
		
		done = next(done_iter)
		obs = next(obs_iter)
		obs = np.concatenate((obs[:-10],obs[-7:]))

		new_path['next_observations'].append(obs)

	path.update(new_path)
	return path

def window_adapt(self,path):
	obs_iter = iter(path['observations']+[path['next_observations'][-1]])
	done_iter = iter(path['terminals'])
	info_iter = iter(path['env_infos'])
	history = deque(np.zeros(self.history_shape),self.history_shape[0])
	is_nonnoop = deque([False]*self.history_shape[0],self.history_shape[0])
	prev_nonnoop = deque(np.zeros(self.nonnoop_shape),self.nonnoop_shape[0])
	new_path = {'observations':[],'next_observations':[]}

	history.append(next(obs_iter))
	if self.include_target:
		obs = np.concatenate((np.ravel(history),np.ravel(prev_nonnoop),np.ravel(path['env_infos'][0]['targets'])))
	else:
		obs = np.concatenate((np.ravel(history),np.ravel(prev_nonnoop),))
	done = False
	while not done:
		new_path['observations'].append(obs)

		if len(history) == self.history_shape[0] and is_nonnoop[0]:
			prev_nonnoop.append(history[0])
		history.append(next(obs_iter))
		info = next(info_iter)
		is_nonnoop.append(info['noop'])
		done = next(done_iter)

		if self.include_target:
			obs = np.concatenate((np.ravel(history),np.ravel(prev_nonnoop),np.ravel(info['targets'])))
		else:
			obs = np.concatenate((np.ravel(history),np.ravel(prev_nonnoop),))
		new_path['next_observations'].append(obs)

	path.update(new_path)
	return path

def action_adapt(self,path):
	done_iter = iter(path['terminals'])
	info_iter = iter(path['env_infos'])
	if self.action_type in ['target','disc_target','cat_target','basis_target','joint']:
		new_path = {'actions':[]}
	else:
		return None

	done = False
	while not done:
		info = next(info_iter)
		done = next(done_iter)

		if self.action_type in ['disc_target','cat_target','basis_target']:
			new_path['actions'].append(1.*F.one_hot(torch.tensor([
				info.get('target_pred',info['target_index'])
				]),len(info['targets'])).float().flatten().numpy())
		elif self.action_type in ['target']:
			new_path['actions'].append(info.get('target_pred',info['targets'][info['target_index']]))
		elif self.action_type in ['joint']:
			new_path['actions'].append(info['joint_action'])

	path.update(new_path)
	return path

def reward_adapt(self,path):
	path['rewards'] = np.maximum(np.minimum(path['rewards'],1),-self.input_penalty)
	# path['rewards'] = np.maximum(np.minimum(path['rewards'],100),0)
	return path

def adapt_factory(base,adapt_funcs):
	class PathAdapter(base):
		def step(self,action):
			obs,r,done,info = super().step(action)
			info['adapt'] = False
			return obs,r,done,info
		def adapt_path(self,path):
			if path['env_infos'][0].get('adapt',True):
				return reduce(lambda value,func:func(self,value), adapt_funcs, path)
			return path
	return PathAdapter

def multiworld_factory(base):
	class multiworld(base):
		def get_env_state(self):
			return None,None
		def set_env_state(self,tuple):
			pass
		def set_to_goal(self,goal):
			pass
		def get_goal(self):
			return None
		def get_image(self, width=None, height=None, camera_name=None):
			return self.env.render(width=width,height=height)
	return multiworld

railrl_class = lambda env_class, adapt_funcs: adapt_factory(env_class,[checkoff_adapt,action_adapt,reward_adapt,*adapt_funcs])

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
class OnOffReplayBuffer(EnvReplayBuffer):
	def __init__(self,max_replay_buffer_size,env,env_info_sizes=None):
		super().__init__(max_replay_buffer_size,env,env_info_sizes)
		self.demos_replay_buffer = EnvReplayBuffer(max_replay_buffer_size,env,env_info_sizes)

	def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
		if kwargs['env_info'].get('offline',False):
			self.demos_replay_buffer.add_sample(observation, action, reward, terminal, next_observation, **kwargs)
		else:
			super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)

	def random_batch(self, batch_size):
		if self._size >= batch_size//2:
			online_batch = super().random_batch(batch_size//2)
			demo_batch = self.demos_replay_buffer.random_batch(batch_size-batch_size//2)
		else:
			online_batch = super().random_batch(self._size)
			demo_batch = self.demos_replay_buffer.random_batch(batch_size-self._size)
		batch = dict(
			observations=np.concatenate((online_batch['observations'],demo_batch['observations'])),
			actions=np.concatenate((online_batch['actions'],demo_batch['actions'])),
			rewards=np.concatenate((online_batch['rewards'],demo_batch['rewards'])),
			terminals=np.concatenate((online_batch['terminals'],demo_batch['terminals'])),
			next_observations=np.concatenate((online_batch['next_observations'],demo_batch['next_observations'])),
		)
		return batch

	def num_steps_can_sample(self):
		return super().num_steps_can_sample() + self.demos_replay_buffer.num_steps_can_sample()

class RecordPathsReplayBuffer(EnvReplayBuffer):
	def __init__(self,max_replay_buffer_size,env,env_info_sizes=None):
		super().__init__(max_replay_buffer_size,env,env_info_sizes)
		self.paths = []

	def add_path(self, path):
		self.paths.append(path)
		super().add_path(path)

	def get_snapshot(self):
		return {'paths': self.paths}

# from discrete_policy import *
# from ent_policy import *
# from railrl.torch.distributions import TanhNormal
# class ScaledTanhNormal(TanhNormal):
# 	def __init__(self, normal_mean, normal_std, low, high, epsilon=1e-6):
# 		super().__init__(normal_mean,normal_std,epsilon=epsilon)
# 		self.coef = high-low
# 		self.low = low

# 	def sample_n(self, n, return_pre_tanh_value=False):
# 		sample,pretanh = super().sample_n(n,return_pre_tanh_value=return_pre_tanh_value)
# 		if return_pre_tanh_value:
# 			return self.coef*sample-self.low, pretanh
# 		else:
# 			return self.coef*sample-self.low

# 	def _log_prob_from_pre_tanh(self, pre_tanh_value):
# 		log_prob = super()._log_prob_from_pre_tanh(pre_tanh_value)
# 		log_prob -= np.log(self.coef)
# 		return log_prob

# 	def rsample_with_pretanh(self):
# 		sample,pretanh = super().rsample_with_pretanh()
# 		return self.coef*sample-self.low, pretanh

# from railrl.torch.sac.policies.gaussian_policy import TanhGaussianPolicy
# import railrl.torch.pytorch_util as ptu
# LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
# class ScaledTanhGaussianPolicy(TanhGaussianPolicy):
# 	def __init__(
# 			self,
# 			hidden_sizes,
# 			action_low=0,
# 			action_high=1,
# 			**kwargs
# 	):
# 		super().__init__(hidden_sizes,**kwargs)
# 		self.high = action_high
# 		self.low = action_low

# 	def forward(self, obs):
# 		h = obs
# 		for i, fc in enumerate(self.fcs):
# 			h = self.hidden_activation(fc(h))
# 		mean = self.last_fc(h)
# 		if self.std is None:
# 			log_std = self.last_fc_log_std(h)
# 			log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
# 			std = torch.exp(log_std)
# 		else:
# 			std = torch.from_numpy(np.array([self.std, ])).float().to(
# 				ptu.device)

# 		return ScaledTanhNormal(mean, std, self.low, self.high,)

# 	def logprob(self, action, mean, std):
# 		tanh_normal = ScaledTanhNormal(mean, std, self.low, self.high,)
# 		log_prob = tanh_normal.log_prob(
# 			action,
# 		)
# 		log_prob = log_prob.sum(dim=1, keepdim=True)
# 		return log_prob
