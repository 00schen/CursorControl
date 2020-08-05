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
		# self.logger.record('success_metric/accuracy', np.mean(base_env.accuracy))
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

"""awac"""
def window_adapt(env,path):
	obs_iter = iter(path['observations']+[path['next_observations'][-1]])
	done_iter = iter(path['terminals'])
	info_iter = iter(path['env_infos'])
	history = deque(np.zeros(env.history_shape),env.history_shape[0])
	is_nonnoop = deque([False]*env.history_shape[0],env.history_shape[0])
	prev_nonnoop = deque(np.zeros(env.nonnoop_shape),env.nonnoop_shape[0])
	new_path = {'observations':[],'next_observations':[]}

	history.append(next(obs_iter))
	obs = np.concatenate((np.ravel(history),np.ravel(prev_nonnoop),))
	done = False
	while not done:
		new_path['observations'].append(obs)

		if len(history) == env.history_shape[0] and is_nonnoop[0]:
			prev_nonnoop.append(history[0])
		history.append(next(obs_iter))
		info = next(info_iter)
		is_nonnoop.append(info['noop'])
		done = next(done_iter)

		obs = np.concatenate((np.ravel(history),np.ravel(prev_nonnoop),))
		new_path['next_observations'].append(obs)

	path.update(new_path)

def awac_path_loader(env,paths,buffer):
	for path in paths:
		window_adapt(env.envs[0],path)
		env.obs_rms.update(np.array(path['observations']))
		for data in zip(path['observations'],path['next_observations'],path['actions'],path['rewards'],path['terminals'],):
			buffer.add(*data)
		
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