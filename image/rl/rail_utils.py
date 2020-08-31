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

def target_adapt(self,path):
	obs_iter = iter(path['observations']+[path['next_observations'][-1]])
	done_iter = iter(path['terminals'])
	info_iter = iter(path['env_infos'])
	new_path = {'observations':[],'next_observations':[]}

	obs = np.concatenate((next(obs_iter),path['env_infos'][0]['targets'][path['env_infos'][0]['target_index']]))
	done = False
	while not done:
		new_path['observations'].append(obs)

		info = next(info_iter)
		done = next(done_iter)

		obs = np.concatenate((next(obs_iter),info['targets'][info['target_index']]))
		new_path['next_observations'].append(obs)

	path.update(new_path)
	return path

def cap_adapt(self,path):
	path['rewards'] = np.minimum(path['rewards'],self.cap)
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
	action_iter = iter(path['actions'])
	if self.action_type in ['target','disc_target','cat_target','basis_target','joint','disc_traj']:
		new_path = {'actions':[]}
	else:
		return None

	done = False
	while not done:
		info = next(info_iter)
		done = next(done_iter)
		action = next(action_iter)

		if self.action_type in ['disc_target','cat_target','basis_target']:
			new_path['actions'].append(1.*F.one_hot(torch.tensor([
				info.get('target_pred',info['target_index'])
				]),len(info['targets'])).float().flatten().numpy())
		elif self.action_type in ['target']:
			new_path['actions'].append(info.get('target_pred',info['targets'][info['target_index']]))
		elif self.action_type in ['joint']:
			new_path['actions'].append(info['joint_action'])
		elif self.action_type in ['disc_traj']:
			traj = np.zeros(6)
			index = np.argmax(np.abs(action))
			index = index*2 + action[index] > 0
			traj[index] = 1
			new_path['actions'].append(traj)

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
