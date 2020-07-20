import numpy as np

import torch as th
from tensorflow import keras
from tensorflow.keras.layers import Dense

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback,CallbackList,CheckpointCallback
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common import logger

import pybullet as p

import os
import pickle

from envs import *
from collections import namedtuple
from tqdm import tqdm,trange

"""Stable Baseline Callbacks"""
class TensorboardCallback(BaseCallback):
	def __init__(self, curriculum=False, verbose=0):
		super().__init__(verbose)
		self.curriculum = curriculum
	def _on_step(self):
		base_env = self.training_env.envs[0]
		env = self.training_env.envs[0].env.env
		self.min_dist = np.minimum(self.min_dist,np.linalg.norm(env.target_pos - env.tool_pos))	
		self.logger.record('success_metric/success_rate', np.mean(base_env.success_count))
		if self.curriculum:
			self.logger.record('success_metric/t', base_env.scheduler.t)
		self.logger.record('success_metric/init_distance', np.linalg.norm(env.target_pos-env.init_pos))
		self.logger.record('success_metric/min_distance', self.min_dist)
		self.logger.record('success_metric/noop_rate', np.mean(base_env.noop_buffer))
		self.logger.record('success_metric/accuracy', np.mean(base_env.accuracy))
		# self.logger.record('success_metric/log_loss', np.mean(base_env.log_loss))

		return True
	def _on_rollout_start(self):
		self.min_dist = np.inf

class NormCheckpointCallback(BaseCallback):
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
			path = os.path.join(self.save_path, f'norm_{self.num_timesteps}_steps')
			self.training_env.save(path)
		return True

"""railrl"""
from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
class PathAdaptLoader(DictToMDPPathLoader):
	def load_path(self, path, replay_buffer, obs_dict=None):
		replay_buffer.env.adapt_path(path)
		super().load_path(path, replay_buffer, obs_dict)

def window_adapt(self,path):
	obs_iter = iter(path['observations']+[path['next_observations'][-1]])
	done_iter = iter(path['terminals'])
	info_iter = iter(path['env_infos'])
	history = deque(np.zeros(self.history_shape),self.history_shape[0])
	is_nonnoop = deque([False]*self.history_shape[0],self.history_shape[0])
	prev_nonnoop = deque(np.zeros(self.nonnoop_shape),self.nonnoop_shape[0])
	new_path = {'observations':[],'next_observations':[]}

	history.append(next(obs_iter))
	obs = np.concatenate((np.ravel(history),np.ravel(prev_nonnoop),))
	done = False
	while not done:
		new_path['observations'].append(obs)

		if len(history) == self.history_shape[0] and is_nonnoop[0]:
			prev_nonnoop.append(history[0])
		history.append(next(obs_iter))
		info = next(info_iter)
		info['adapt'] = False
		is_nonnoop.append(info['noop'])
		done = next(done_iter)

		obs = np.concatenate((np.ravel(history),np.ravel(prev_nonnoop),))
		new_path['next_observations'].append(obs)
	
	path.update(new_path)

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

from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths
def logger_factory(base):
	class StatsLogger(base):
		def get_diagnostics(self,paths):
			statistics = OrderedDict()

			"""success"""
			success_per_step = get_stat_in_paths(paths, 'env_infos', 'task_success')
			success_per_ep = [np.count_nonzero(s) > 0 for s in success_per_step]
			statistics.update(create_stats_ordered_dict('success',success_per_ep,exclude_max_min=True,))	
			
			"""distance"""
			distance_per_step = get_stat_in_paths(paths, 'env_infos', 'distance_to_target')
			min_distance = [np.amin(s) for s in distance_per_step]
			init_distance = [s[0] for s in distance_per_step]
			final_distance = [s[-1] for s in distance_per_step]
			statistics.update(create_stats_ordered_dict('min_distance',min_distance,))
			statistics['init_distance'] = np.mean(init_distance)
			statistics['final_distance'] = np.mean(final_distance)

			"""cos_error"""
			cos_error_per_step = get_stat_in_paths(paths, 'env_infos', 'cos_error')
			statistics.update(create_stats_ordered_dict('cos_error',cos_error_per_step,))

			"""noop"""
			noop_per_step = get_stat_in_paths(paths, 'env_infos', 'noop')
			statistics.update(create_stats_ordered_dict('noop',noop_per_step,exclude_max_min=True,))

			return statistics
	return StatsLogger

railrl_class = lambda env_name, adapt_funcs: adapt_factory(logger_factory(default_class(env_name)),adapt_funcs)
	
"""Miscellaneous"""
ROLLOUT  = 1
NORM = 2
SIZES2 = 3
SIZES5 = 4
SIZES8 = 5
BLANK = 6
OBS2 = 7
OBS1 = 8
SAMPLE_NAME = {
	ROLLOUT: 'rollout',
	NORM: 'norm',
	SIZES2: 'sizes2',
	SIZES5: 'sizes5',
	SIZES8: 'sizes8',
	BLANK: 'blank',
	OBS2: 'obs2',
	OBS1: 'obs1',
}

class IKAgent:
	def __init__(self,env):
		self.env = env
		self.new_pos = -1
		self.tool_pos = -1
		self.tool = -1
	
	def predict(self,obs):
		joint_states = p.getJointStates(self.env.robot, jointIndices=self.env.robot_left_arm_joint_indices, physicsClientId=self.env.id)
		joint_positions = np.array([x[0] for x in joint_states])

		link_pos = p.getLinkState(self.env.robot, 13, computeForwardKinematics=True, physicsClientId=self.env.id)[0]
		new_pos = p.multiplyTransforms(self.env.target_pos, [0, 0, 0, 1],
					 link_pos-self.env.tool_pos, [0, 0, 0, 1], physicsClientId=self.env.id)[0]

		# new_pos = self.env.target_pos
		new_joint_positions = np.array(p.calculateInverseKinematics(self.env.robot, 13, new_pos, physicsClientId=self.env.id))
		new_joint_positions = new_joint_positions[:7]
		action = 2*(new_joint_positions - joint_positions)

		# p.removeBody(self.new_pos)
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=self.env.id)
		# self.new_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
		# 		basePosition=new_pos, useMaximalCoordinates=False, physicsClientId=self.env.id)

		# p.removeBody(self.tool_pos)
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=self.env.id)
		# self.tool_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
		# 		basePosition=self.env.tool_pos, useMaximalCoordinates=False, physicsClientId=self.env.id)

		return action

class KeyboardAgent:
	key_mappings = {
		'a':'left',
		'd':'right',
		's':'backward',
		'w':'forward',
		'z':'down',
		'x':'up',
	}
	def predict(self,obs):
		key = input('direction: ')
		if key in self.key_mappings:
			return self.key_mappings[key]
		elif key == 'reset':
			return key
		else:
			return None