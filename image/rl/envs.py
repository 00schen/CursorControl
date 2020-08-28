import numpy as np
from numpy.random import default_rng
rng = default_rng()
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from scipy.stats import truncexpon
from scipy.special import softmax
from types import MethodType

import torch
import tensorflow as tf
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.optimizers import Adam

import pybullet as p
import assistive_gym
from gym import spaces,Env
# from stable_baselines3.sac import SAC
# from stable_baselines3.common.running_mean_std import RunningMeanStd

from functools import reduce
from collections import deque
import os

parentname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class IKPretrain:
	def __init__(self,config):
		self.new_pos = -1
		self.tool_pos = -1
	
	def predict(self,env,pred):
		joint_states = p.getJointStates(env.robot, jointIndices=env.robot_left_arm_joint_indices, physicsClientId=env.id)
		joint_positions = np.array([x[0] for x in joint_states])

		link_pos = p.getLinkState(env.robot, 13, computeForwardKinematics=True, physicsClientId=env.id)[0]
		# new_pos = p.multiplyTransforms(pred, [0, 0, 0, 1],
		# 			 link_pos-env.tool_pos, [0, 0, 0, 1], physicsClientId=env.id)[0]
		new_pos = np.array(pred)+np.array(link_pos)-env.tool_pos

		new_joint_positions = np.array(p.calculateInverseKinematics(env.robot, 13, new_pos, physicsClientId=env.id))
		new_joint_positions = new_joint_positions[:7]
		action = new_joint_positions - joint_positions

		clip_by_norm = lambda traj,limit: traj/max(1e-4,norm(traj))*np.clip(norm(traj),None,limit)
		action = clip_by_norm(action,.1)

		# p.removeBody(self.new_pos)
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[10, 255, 10, 1], physicsClientId=env.id)
		# self.new_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
		# 		basePosition=pred, useMaximalCoordinates=False, physicsClientId=env.id)

		# p.removeBody(self.tool_pos)
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=env.id)
		# self.tool_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
		# 		basePosition=env.tool_pos, useMaximalCoordinates=False, physicsClientId=env.id)

		return action

def sparse_factory(env_name):
	class Sparse(Env):
		def __init__(self,config):
			self.env = {
				"ScratchItch": assistive_gym.ScratchItchJacoEnv,
				"Feeding": assistive_gym.FeedingJacoEnv,
				"Laptop": assistive_gym.LaptopJacoEnv,
				"LightSwitch": assistive_gym.LightSwitchJacoEnv,
				"Reach": assistive_gym.ReachJacoEnv,
			}[env_name](**config['env_kwargs'])

			self.timesteps = 0
			self.step_limit = config['step_limit']
			self.success_count = deque([0]*20,20)
			self.end_early = config['end_early']
			self.reward_type = config['reward_type']
			self.phi = config['phi']

		def step(self,action):
			obs,r,done,info = self.env.step(action)
			self.timesteps += 1
			if norm(self.env.target_pos-self.env.tool_pos) < .025:
				self.env.task_success += 1
				info['task_success'] = True

			if not self.end_early:
				if self.timesteps >= self.step_limit:
					done = True
					self.success_count.append(self.env.task_success > 0)
				else:
					done = False
					r = 100*info['task_success']
			else:
				r = (self.env.task_success > 0)
				if self.env.task_success > 0 or self.timesteps >= self.step_limit:
					done = True
					self.success_count.append(self.env.task_success > 0)
				else:
					done = False

			r += self.phi({
				'distance_to_target': info['distance_to_target'],
				# 'diff_distance': info['diff_distance']
			}[self.reward_type])

			# info['target_pos'] = self.env.target_pos
			info['target_index'] = self.env.target_index
			info['targets'] = self.env.targets

			return obs,r,done,info

		def reset(self):
			obs = self.env.reset()
			self.timesteps = 0
			return obs

		def render(self,mode=None,**kwargs):
			return self.env.render(mode)
		def seed(self,value):
			self.env.seed(value)
		def close(self):
			self.env.close()

	return Sparse

class Noise():
	def __init__(self, action_space=None, sd=.1, dropout=.3, lag=.85, include=(0,1,2)):
		self.sd = sd
		self.lag = lag
		self.dropout = dropout

		if action_space is not None:
			self.action_space = action_space
			self.dim = action_space.shape[0]
		self.noises = [[self._add_awg,self._add_lag,self._add_dropout][i] for i in include]+[lambda value: value]

	def reset(self):
		# self.noise = rng.normal(np.identity(self.dim), self.sd)
		self.lag_buffer = deque([],10)

	def _add_awg(self, action):
		# return action@self.noise
		return action + rng.normal(np.zeros(action.shape), self.sd)
	def _add_lag(self, action):
		self.lag_buffer.append(action)
		return np.array(self.lag_buffer.popleft() if rng.random() > self.lag else self.lag_buffer[0])
	def _add_dropout(self, action):
		return np.array(action if rng.random() > self.dropout else self.action_space.sample())
	def __call__(self,action):
		return reduce(lambda value,func: func(value), self.noises, action)

def shared_autonomy_factory(base):
	class SharedAutonomy(base):
		def __init__(self,config):
			super().__init__(config)
			self.noop_buffer = deque([],2000)
			self.input_penalty = config['input_penalty']
			self.oracle_size = config['oracle_size']
			self.action_penalty = config['action_penalty']
			self.action_type = config['action_type']
			
			def trajectory(obs,info):
				recommend = np.zeros(config['oracle_size'])
				recommend[0:3] = self.env.target_pos - self.env.tool_pos
				return recommend
			def dist_discrete_traj(obs,info):
				"""User corrects direction robot is most off course"""
				recommend = np.zeros(config['oracle_size'])
				if info['diff_distance'] < -.1 or info['distance_to_target'] > .5:
					traj = self.env.target_pos-self.env.tool_pos
					axis = np.argmax(np.abs(traj))
					recommend[2*axis+(traj[axis]>0)] = 1
				return recommend
			def rad_discrete_traj(obs,info):
				"""if the robot is off course, user corrects the most off direction"""
				recommend = np.zeros(config['oracle_size'])
				# t = np.minimum(info['distance_to_target'],1)
				# threshold = t*config['threshold'] + (1-t)
				threshold = config['threshold']
				if info['cos_error'] < threshold:
					traj = self.env.target_pos-self.env.tool_pos
					axis = np.argmax(np.abs(traj))
					recommend[2*axis+(traj[axis]>0)] = 1
				return recommend
			def user(obs,info):
				print("function called")
				recommend = self.get_user_input(obs,info)
				return recommend
			def user_model(obs,info):
				if self.prev_noop is None or not self.prev_noop:
					prob = .15*(1-info['cos_error'])
				else:
					prob = .6 if info['cos_error'] < .25 else .3
				recommend = np.zeros(config['oracle_size'])
				if rng.random() < prob:
					traj = self.env.target_pos-self.env.tool_pos
					axis = np.argmax(np.abs(traj))
					recommend[2*axis+(traj[axis]>0)] = 1

				self.prev_noop = not np.count_nonzero(recommend)
				return recommend
			def dd_target(obs,info):
				# diff_distances = np.array([norm(self.env.tool_pos-target_pos) for target_pos in self.env.targets])\
				# 				- np.array([norm(info['old_tool_pos']-target_pos) for target_pos in self.env.targets])
				diff_distances = np.array([-norm(self.env.tool_pos-target_pos) for target_pos in self.env.targets])
				recommend = np.zeros(config['oracle_size'])
				if self.env.target_index != np.argmax(diff_distances):
					# traj = self.env.target_pos-self.env.targets[np.argmax(diff_distances)]
					traj = self.env.target_pos-self.env.tool_pos
					axis = np.argmax(np.abs(traj))
					recommend[2*axis+(traj[axis]>0)] = 1
				return recommend
			def ded_target(obs,info):
				recommend = np.zeros(config['oracle_size'])
				traj = self.env.target_pos-self.env.tool_pos
				axis = np.argmax(np.abs(traj))
				recommend[2*axis+(traj[axis]>0)] = 1
				return recommend
			def random_traj(obs,info):
				recommend = np.zeros(config['oracle_size'])
				abs_traj = np.abs(self.env.target_pos-self.env.tool_pos)
				indices = np.argsort(abs_traj)
				if abs_traj[indices[2]] - abs_traj[indices[1]] > config['threshold']\
					or info['distance_to_target'] < .1:
					recommend = np.zeros(config['oracle_size'])
					traj = self.env.target_pos-self.env.tool_pos
					axis = np.argmax(np.abs(traj))
					recommend[2*axis+(traj[axis]>0)] = 1
				return recommend
			self.determiner = {
				# 'target': lambda obs,info: self.env.target_pos,
				# 'trajectory': trajectory,
				# 'discrete_target': lambda obs,info: self.env.target_num,
				# 'dist_hot_cold': dist_hot_cold,
				# 'rad_hot_cold': rad_hot_cold,
				# 'dist_discrete_traj': dist_discrete_traj,
				# 'rad_discrete_traj': rad_discrete_traj,
				'user': user,
				'user_model': user_model,
				# 'dd_target': dd_target,
				# 'ded_target': ded_target,
				# 'random_traj': random_traj,
			}[config['oracle']]

			if config['action_type'] in ['target', 'trajectory', 'disc_target','cat_target','basis_target']:
				self.pretrain = config['pretrain'](config)
			joint = lambda action: action
			target = lambda pred: self.pretrain.predict(self.env,pred)
			trajectory = lambda traj: target(self.env.tool_pos+traj)
			def cat_target(index):
				index = np.argmax(index)
				return target(self.env.targets[index])
			def disc_traj(index):
				index = np.argmax(index)
				traj = 
			self.translate = {
				# 'target': target,
				'trajectory': trajectory,
				'joint': joint,
				# 'cat_target': cat_target,
			}[config['action_type']]
			self.action_space = {
				# "target": spaces.Box(-1,1,(3,)),
				"trajectory": spaces.Box(-1,1,(3,)),
				"joint": spaces.Box(-1,1,(7,)),
				# "cat_target": spaces.Box(0,1,(self.env.num_targets,)),
				"disc_traj": spaces.Box(0,1,(6,)),
			}[config['action_type']]

		def step(self,action):
			t_action = self.translate(action)
			obs,r,done,info = super().step(t_action)
			info['joint_action'] = t_action
			obs = self.predict(obs,info)
			r -= self.input_penalty*(not info['noop'])
			# r -= self.action_penalty*cosine(self.env.tool_pos-info['old_tool_pos'],self.coasted_input)

			if self.action_type in ['disc_target','cat_target']:
				info['accuracy'] = np.argmax(action) == self.env.target_index
				info['pred_target_dist'] = norm(self.env.targets[np.argmax(action)]-self.env.target_pos)

				# p.removeBody(self.pred_target)
				# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[10, 255, 10, 1])
				# self.pred_target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
				# 		basePosition=self.env.targets[np.argmax(action)], useMaximalCoordinates=False)

			elif self.action_type in ['basis_target']:
				info['pred_target_dist'] = norm(np.sum(self.env.targets*action.reshape((-1,1)),axis=0)-self.env.target_pos)
			elif self.action_type in ['target']:
				info['pred_target_dist'] = norm(action-self.env.target_pos)

			return obs,r,done,info

		def reset(self):
			self.pred_target = -1
			self.prev_noop = None
			obs = super().reset()
			obs = np.concatenate((obs,np.zeros(self.oracle_size)))
			return obs

		def predict(self,obs,info):
			recommend = self.determiner(obs,info)

			info['recommend'] = recommend
			info['noop'] = not np.count_nonzero(recommend)
			self.noop_buffer.append(info['noop'])

			return np.concatenate((obs,recommend))

	return SharedAutonomy

def window_factory(base):
	class PrevNnonNoopK(base):
		def __init__(self,config):
			super().__init__(config)
			self.include_target = config['include_target']
			self.current_obs_dim = config['obs_size']+config['oracle_size']
			self.history_shape = (config['num_obs'],self.current_obs_dim)
			self.nonnoop_shape = (config['num_nonnoop'],self.current_obs_dim)
			if self.include_target:
				self.observation_space = spaces.Box(-np.inf,np.inf,(np.prod(self.history_shape)\
													+np.prod(self.nonnoop_shape)+3*self.env.num_targets,))
			else:
				self.observation_space = spaces.Box(-np.inf,np.inf,(np.prod(self.history_shape)+np.prod(self.nonnoop_shape),))

		def step(self,action):
			obs,r,done,info = super().step(action)

			if len(self.history) == self.history_shape[0] and self.is_nonnoop[0]:
				self.prev_nonnoop.append(self.history[0])

			self.history.appendleft(obs)
			self.is_nonnoop.appendleft((not info['noop']))
			info['current_obs'] = obs
			
			if self.include_target:
				return np.concatenate((*self.history,*self.prev_nonnoop,*self.env.targets)),r,done,info
			else:
				return np.concatenate((*self.history,*self.prev_nonnoop,)),r,done,info

		def reset(self):
			obs = super().reset()
			if obs is None:
				return self.observation_space.sample()
			self.history = deque(np.zeros(self.history_shape),self.history_shape[0])
			self.is_nonnoop = deque([False]*self.history_shape[0],self.history_shape[0])
			self.prev_nonnoop = deque(np.zeros(self.nonnoop_shape),self.nonnoop_shape[0])
			self.history.appendleft(obs)

			if self.include_target:
				return np.concatenate((*self.history,*self.prev_nonnoop,*self.env.targets))
			else:
				return np.concatenate((*self.history,*self.prev_nonnoop,))
	return PrevNnonNoopK

import pygame as pg
def feedback_factory(base):
	SCREEN_SIZE = 250
	class Feedback(base):
		def __init__(self,config):
			super().__init__(config)
			self.screen = None

		def get_user_input(self,obs,info):
			recommend = np.zeros(6)
			keys = p.getKeyboardEvents()
			inputs = {
				p.B3G_LEFT_ARROW: 	np.array([0,1,0,0,0,0]),
				p.B3G_RIGHT_ARROW: 	np.array([1,0,0,0,0,0]),
				ord('e'):		 	np.array([0,0,1,0,0,0]),
				ord('d'):		 	np.array([0,0,0,1,0,0]),
				p.B3G_UP_ARROW:		np.array([0,0,0,0,0,1]),
				p.B3G_DOWN_ARROW:	np.array([0,0,0,0,1,0])
			}
			prints = {
				p.B3G_LEFT_ARROW: 	'left',
				p.B3G_RIGHT_ARROW: 	'right',
				ord('e'):		 	'forward',
				ord('d'):		 	'backward',
				p.B3G_UP_ARROW:		'up',
				p.B3G_DOWN_ARROW:	'down'
			}
			for key in inputs:
				if key in keys and keys[key]&p.KEY_WAS_TRIGGERED:
					recommend = inputs[key]
					label = prints[key]
					print(label)
			if not np.count_nonzero(recommend):
				label = 'noop'
				print('noop')

			if self.screen is not None:
				self.screen.fill(pg.color.THECOLORS["white"])
				font = pg.font.Font(None, 24)
				text = label
				text = font.render(text, 1, pg.color.THECOLORS["black"])
				self.screen.blit(text, (125,125))
			return recommend

		def render(self,mode=None):
			if mode == 'human':
				pg.init()
				self.screen = pg.display.set_mode((SCREEN_SIZE,SCREEN_SIZE)) 
			return super().render(mode)
	return Feedback

def sanity_factory(base):
	class AddTargetIndex(base):
		def __init__(self,config):
			super().__init__(config)
			self.observation_space = spaces.Box(-np.inf,np.inf,(self.observation_space.shape[0]+self.env.num_targets,))
		def step(self,action):
			obs,r,done,info = super().step(action)
			targets = np.zeros(len(self.env.targets))
			targets[self.env.target_index] = 1
			obs = np.concatenate((obs,targets))
			return obs,r,done,info
		def reset(self):
			obs = super().reset()
			targets = np.zeros(len(self.env.targets))
			targets[self.env.target_index] = 1
			obs = np.concatenate((obs,targets))
			return obs
	return AddTargetIndex

def new_target_factory(base):
	class NewTarget(base):
		def __init__(self,config):
			super().__init__(config)
			self.num_episodes = 0
			self.delay = config['target_delay']
		def reset(self):
			def set_target(self):
				self.target_pos = .5*(self.targets[0]+self.targets[-1])
				return self.target_pos
			self.num_episodes += 1
			if self.num_episodes > self.delay:		
				self.env.set_target = MethodType(set_target,self.env)
			obs = super().reset()
			print(self.env.target_pos,self.env.targets)
			print(self.num_episodes)
			return obs
	return NewTarget

class CurriculumScheduler:
	def __init__(self,config):
		self.t = config['init_t']
		self.step_count = 0
		self.increment = config['curr_inc']
		self.phase = {
			'linear': lambda: self.t + self.increment,
			'concave': lambda: self.t + self.increment**self.step_count,
			'convex': lambda: self.t + self.increment**(-self.step_count)
		}[config['curr_phase']]
	
	def step(self):
		self.step_count += 1
		self.t = min(self.phase(), 1)

	def get_t(self):
		return self.t if rng.random() < .95 else rng.random()*self.t

def moving_init_factory(base):
	class MovingInit(base):
		def __init__(self,config):
			super().__init__(config)
			self.wait_time = 0
			config.update({'init_t': .3, 'curr_inc': .03, 'curr_phase': 'linear'})
			self.scheduler = CurriculumScheduler(config)

		def reset(self):
			self.wait_time += 1
			if self.wait_time > 20 and np.mean(self.success_count) > .5:
				self.wait_time = 0
				self.scheduler.step()
			t = self.scheduler.get_t()
			def init_start_pos(self,og_init_pos):
				nonlocal t
				return t*og_init_pos + (1-t)*self.target_pos
			self.env.init_start_pos = MethodType(init_start_pos,self.env)
			obs = super().reset()
			return obs
	return MovingInit

default_config = {
	'step_limit': 200,
	'end_early': True,
	'env_kwargs': {},
	'oracle_size': 6,
	'reward_type': 'distance_to_target',
	'phi': lambda d: 0,
	'noise': False,
}
env_keys = ('env_name','obs_size','pretrain')
env_map = {
	"ScratchItch": dict(zip(env_keys,("ScratchItch",24,IKPretrain))),
	"Feeding": dict(zip(env_keys,("Feeding",22,IKPretrain))),
	# "Laptop": dict(zip(env_keys,("Laptop",15,IKPretrain))),
	# "LightSwitch": dict(zip(env_keys,("LightSwitch",15,IKPretrain))),
	"Laptop": dict(zip(env_keys,("Laptop",18,IKPretrain))),
	"LightSwitch": dict(zip(env_keys,("LightSwitch",15,IKPretrain))),
	"Reach": dict(zip(env_keys,("Reach",14,IKPretrain))),
	}
default_class = lambda env_name: reduce(lambda value,func: func(value),
				[sparse_factory,shared_autonomy_factory,window_factory],env_name)