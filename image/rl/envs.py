import numpy as np
import numpy.random as random
from numpy.linalg import norm
from scipy.spatial.distance import cosine
from types import MethodType

import torch
import tensorflow as tf
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.optimizers import Adam

import pybullet as p
import assistive_gym
from gym import spaces,Env
from stable_baselines3.sac import SAC
from stable_baselines3.common.running_mean_std import RunningMeanStd

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
		new_pos = pred+link_pos-env.tool_pos
		# print(pred, env.tool_pos)

		new_joint_positions = np.array(p.calculateInverseKinematics(env.robot, 13, new_pos, physicsClientId=env.id))
		new_joint_positions = new_joint_positions[:7]
		action = new_joint_positions - joint_positions

		# p.removeBody(self.new_pos)
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=env.id)
		# self.new_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
		# 		basePosition=new_pos, useMaximalCoordinates=False, physicsClientId=env.id)

		# p.removeBody(self.tool_pos)
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=env.id)
		# self.tool_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
		# 		basePosition=env.tool_pos, useMaximalCoordinates=False, physicsClientId=env.id)

		return action

class GymWrapper(Env):
	def __init__(self,config):
		self.env = {
			"ScratchItch": assistive_gym.ScratchItchJacoEnv,
			"Feeding": assistive_gym.FeedingJacoEnv,
			"Laptop": assistive_gym.LaptopJacoEnv,
			"LightSwitch": assistive_gym.LightSwitchJacoEnv,
			"Reach": assistive_gym.ReachJacoEnv,
		}[config['env_name']](**config['env_kwargs'])

		if config['action_type'] in ['target', 'trajectory']:
			self.pretrain = config['pretrain'](config)
		joint_action = lambda action: action
		target_action = lambda pred: self.pretrain.predict(self.env,pred)
		clip_by_norm = lambda traj,limit: traj/norm(traj)*np.clip(norm(traj),None,limit)
		trajectory_action = lambda traj: target_action(self.env.tool_pos+clip_by_norm(traj,.05))
		self.translate_action = {
			'joint': joint_action,
			'target': target_action,
			'trajectory': trajectory_action,
		}[config['action_type']]
		self.action_space = config['action_space']

	def step(self,action):
		obs,r,done,info = self.env.step(self.translate_action(action))
		return obs,r,done,info

	def reset(self):
		obs = self.env.reset()
		return obs

	def render(self,mode=None,**kwargs):
		self.env.render()
	def seed(self,value):
		self.env.seed(value)
	def close(self):
		self.env.close()

class Noise():
	def __init__(self, action_space, sd=.1, dropout=.3, lag=.85, include=(0,1,2)):
		self.sd = sd
		self.dim = action_space.shape[0]
		self.lag = lag
		self.dropout = dropout

		random.seed(12345)

		self.action_space = action_space
		self.noises = [[self._add_awg,self._add_lag,self._add_dropout][i] for i in include]+[lambda value: value]

	def reset(self):
		# self.noise = random.normal(np.identity(self.dim), self.sd)
		self.lag_buffer = deque([],10)

	def _add_awg(self, action):
		# return action@self.noise
		return action + random.normal(np.zeros(action.shape), self.sd)
	def _add_dropout(self, action):
		return np.array(action if random.random() > self.dropout else self.action_space.sample())
	def _add_lag(self, action):
		self.lag_buffer.append(action)
		return np.array(self.lag_buffer.popleft() if random.random() > self.lag else self.lag_buffer[0])
	def __call__(self,action):
		return reduce(lambda value,func: func(value), self.noises, action)

class SharedAutonomy(GymWrapper):
	def __init__(self,config):
		super().__init__(config)
		self.indices = config['indices']
		self.noop_buffer = deque([],2000)
		
		def dist_hot_cold(obs,info):
			"""User only tells if the robot is moving away from target or very off"""
			recommend = np.zeros(10)
			if info['diff_distance'] < -.1 or info['distance_target'] > .5:
				recommend[0] = 1
			return recommend
		def rad_hot_cold(obs,info):
			"""User only tells if the robot is off course"""
			recommend = np.zeros(10)
			if info['cos_off_course'] < .85:
				recommend[0] = 1
			return recommend
		def dist_discrete_traj(obs,info):
			"""if the robot is moving away from target or very off, user corrects the most off direction"""
			recommend = np.zeros(10)
			if info['diff_distance'] < -.1 or info['distance_target'] > .5:
				traj = self.env.target_pos-self.env.tool_pos
				axis = np.argmax(np.abs(traj))
				recommend[2*axis+(traj[axis]>0)] = 1
			return recommend
		def rad_discrete_traj(obs,info):
			"""if the robot is off course, user corrects the most off direction"""
			recommend = np.zeros(10)
			if info['cos_off_course'] < .85:
				traj = self.env.target_pos-self.env.tool_pos
				axis = np.argmax(np.abs(traj))
				recommend[2*axis+(traj[axis]>0)] = 1
			return recommend
		self.determiner = {
			'target': lambda obs,info: self.env.target_pos,
			'trajectory': lambda obs,info: self.env.target_pos - self.env.tool_pos,
			'discrete_target': lambda obs,info: self.env.target_num,
			'dist_hot_cold': dist_hot_cold,
			'rad_hot_cold': rad_hot_cold,
			'dist_discrete_traj': dist_discrete_traj,
			'rad_discrete_traj': rad_discrete_traj,
		}[config['oracle']]

		if not config['noise']:
			self.noise = Noise(spaces.Box(-.01,.01,(config['oracle_size'],)),include=())
		else:
			self.noise = {
				'target': Noise(spaces.Box(-.1,.1,(config['oracle_size'],))),
				'trajectory': Noise(spaces.Box(-.01,.01,(config['oracle_size'],))),
				'discrete': lambda: self.env.target_num,
			}[config['oracle']]

	def step(self,action):
		obs,r,done,info = super().step(action)
		obs = self.predict(obs,info)
		return obs,r,done,info

	def reset(self):
		obs = super().reset()
		self.noise.reset()
		obs = self.predict(obs,{'diff_distance':0,
								'distance_target':norm(self.env.target_pos-self.env.tool_pos),
								'cos_off_course': 1})
		return obs

	def predict(self,obs,info):
		recommend = self.noise(self.determiner(obs,info))
		self.noop_buffer.append(not np.count_nonzero(recommend))
		return np.concatenate((obs[self.indices[0]],obs[self.indices[1]],recommend))

class Sparse(SharedAutonomy):
	def __init__(self,config):
		super().__init__(config)
		self.timesteps = 0
		self.step_limit = config['step_limit']
		self.success_count = deque([0]*20,20)
		self.end_early = config['end_early']
		self.reward_type = config['reward_type']
		self.phi = config['phi']

	def step(self,action):
		obs,r,done,info = super().step(action)
		self.timesteps += 1

		if not self.end_early:
			if self.timesteps >= self.step_limit:
				done = True
				self.success_count.append(self.env.task_success > 0)
			else:
				done = False
				r = 100.0*info['task_success']
		else:
			r = 100.0*(self.env.task_success > 0)
			if self.env.task_success > 0 or self.timesteps >= self.step_limit:
				done = True
				self.success_count.append(self.env.task_success > 0)
			else:
				done = False

		r += self.phi({
			'distance_target': info['distance_target'],
			'diff_distance': info['diff_distance']
		}[self.reward_type])

		return obs,r,done,info

	def reset(self):
		obs = super().reset()
		self.timesteps = 0
		return obs

class PreviousN(Sparse):
	def __init__(self,config):
		super().__init__(config)
		self.history_shape = (config['num_obs'],config['sa_obs_size']+config['oracle_size'])
		self.observation_space = spaces.Box(-np.inf,np.inf,(np.prod(self.history_shape),))

	def step(self,action):
		obs,r,done,info = super().step(action)
		self.history.append(obs)
		return np.ravel(self.history),r,done,info

	def reset(self):
		obs = super().reset()
		self.history = deque(np.zeros(self.history_shape),self.history_shape[0])
		self.history.append(obs)
		return np.ravel(self.history)

class MovingInit(PreviousN):
	def __init__(self,config):
		super().__init__(config)
		self.t = .01
		self.wait_time = 0

	def reset(self):
		self.wait_time += 1
		if self.wait_time > 20 and np.mean(self.success_count) > .5:
			self.t = min(self.t + .01, 1)
			self.wait_time = 0
		t = self.t
		def init_start_pos(self,og_init_pos):
			nonlocal t
			return t*og_init_pos + (1-t)*self.target_pos
		self.env.init_start_pos = MethodType(init_start_pos,self.env)
		obs = super().reset()
		return obs

class TargetRegion(PreviousN):
	def __init__(self,config):
		super().__init__(config)
		self.t = .05
		self.wait_time = 0

	def reset(self):
		self.wait_time += 1
		if self.wait_time > 20 and np.mean(self.success_count) > .5:
			self.t = min(self.t + .005, 1)
			self.wait_time = 0
		t = self.t

		def generate_targets(self):
			nonlocal t
			lim = (np.clip(self.init_pos-t*2*np.array([.75,.5,.35]),(-1,-1,.5),(.5,0,1.2)),
				np.clip(self.init_pos+t*2*np.array([.75,.5,.35]),(-1,-1,.5),(.5,0,1.2)))
			target_pos = self.target_pos = self.np_random.uniform(*lim)
			if self.gui:
				sphere_collision = -1
				sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
				self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual,
												basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)
		self.env.generate_targets = MethodType(generate_targets,self.env)

		obs = super().reset()
		return obs

default_config = {
	'step_limit': 200,
	'end_early': False,
	'env_kwargs': {},
	'oracle_size': 3,
	'num_obs': 5,
	'reward_type': 'distance_target',
	'phi': lambda d: 1/d,
	'noise': False,
	}
env_keys = ('env_name','obs_size','sa_obs_size','indices','pretrain')
env_map = {
	"ScratchItch": dict(zip(env_keys,("ScratchItch",30,24,(slice(7),slice(13,30)),IKPretrain))),
	"Feeding": dict(zip(env_keys,("Feeding",25,22,(slice(7),slice(10,25)),IKPretrain))),
	"Laptop": dict(zip(env_keys,("Laptop",21,18,(slice(7),slice(10,21)),IKPretrain))),
	"LightSwitch": dict(zip(env_keys,("LightSwitch",21,18,(slice(7),slice(10,21)),IKPretrain))),
	"Reach": dict(zip(env_keys,("Reach",17,14,(slice(7),slice(10,17)),IKPretrain))),
	}
action_keys= ('action_type','action_space')
action_map = {
	"target": dict(zip(action_keys,("target", spaces.Box(-1,1,(3,))))),
	"trajectory": dict(zip(action_keys,("trajectory", spaces.Box(-1,1,(3,))))),
	"joint": dict(zip(action_keys,("joint", spaces.Box(-1,1,(7,)))))
	}
