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
	
	def predict(self,env,pred,obs):
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
		}[env_name](**config['env_kwargs'])

		if config['action_type'] in ['target', 'trajectory']:
			self.pretrain = config['pretrain'](config)
		joint_action = lambda action: action
		target_action = lambda pred: self.pretrain.predict(self.env,pred)
		trajectory_action = lambda traj: target_action(self.env.tool_pos+tf.clip_by_norm(traj,.05).numpy())
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
	def __init__(self, action_space, sd=.1, dropout=.3, lag=.85, batch=1, include=(0,1,2)):
		self.sd = sd
		self.dim = action_space.shape[0]
		self.lag = lag
		self.dropout = dropout

		self.action_space = action_space
		self.noises = [[self._add_awg,self._add_lag,self._add_dropout][i] for i in include]

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

class Oracle:
	def __init__(self,config):
		self.determiner = config['determiner']
		self.indices = config['indices']
		self.noise = Noise(spaces.Box(-.01,.01,(config['oracle_size'],)))
		self.reset()

	def reset(self):
		self.noise.reset()

	def predict(self,obs):
		recomend = self.noise(self.determiner())
		return np.concatenate((obs[self.indices[0]],obs[self.indices[1]],recomend))

class SharedAutonomy(GymWrapper):
	def __init__(self,config):
		super().__init__(config)
		config['determiner'] = {
			'target': lambda: self.env.target_pos,
			'trajectory': lambda: self.env.target_pos - self.env.tool_pos,
			'discrete': lambda: self.env.target_num,
		}[config['oracle']]
		self.oracle = Oracle(config)

	def step(self,action):
		obs,r,done,info = super().step(action)
		obs = self.oracle.predict(obs)
		return obs,r,done,info

	def reset(self):
		obs = super().reset()
		self.oracle.reset()
		obs = self.oracle.predict(obs)
		return obs

class Sparse(SharedAutonomy):
	def __init__(self,config):
		super().__init__(config)
		self.timesteps = 0
		self.step_limit = config['step_limit']
		self.success_count = deque([0]*20,20)
		self.end_early = config['end_early']

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

		r -= info['distance_target']

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
		self.t = .1
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
			self.t = min(self.t + .01, 1)
			self.wait_time = 0
		t = self.t

		# def init_start_pos(self,og_init_pos):
			# return np.array([-.3,-.6,.85])
		# self.env.init_start_pos = MethodType(init_start_pos,self.env)
		# def generate_targets(self):
		# 	nonlocal t
		# 	target_pos = self.target_pos = self.init_pos + t*np.array([.9,.6,.35])*self.np_random.uniform(-1,1,3)
		# 	if self.gui:
		# 		sphere_collision = -1
		# 		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		# 		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual,
		# 										basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)
		# self.env.generate_targets = MethodType(generate_targets,self.env)

		def generate_targets(self):
			nonlocal t
			lim = (np.clip(self.init_pos-t*np.array([.75,.5,.35]),(-1,-1,.5),(.5,0,1.2)),
				np.clip(self.init_pos+t*np.array([.75,.5,.35]),(-1,-1,.5),(.5,0,1.2)))
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
	'radius_max': 1,
	'pred_step': 10,
	'step_limit': 200,
	'action_gamma': 10,
	'coop': False,
	'end_early': False,
	'env_kwargs': {},
	'oracle_size': 3,
	'num_obs': 5
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
