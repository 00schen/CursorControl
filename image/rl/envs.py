import numpy as np
from numpy.random import default_rng
rng = default_rng()
from numpy.linalg import norm
from types import MethodType

import pybullet as p
import assistive_gym
from gym import spaces,Env

from functools import reduce
from collections import deque
import os

parentname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class IKAgent:
	def __init__(self,config):
		self.new_pos = -1
		self.tool_pos = -1
		self.action_clip = config['action_clip']

	def predict(self,env,pred):
		joint_states = p.getJointStates(env.robot, jointIndices=env.robot_left_arm_joint_indices, physicsClientId=env.id)
		joint_positions = np.array([x[0] for x in joint_states])

		link_pos = p.getLinkState(env.robot, 13, computeForwardKinematics=True, physicsClientId=env.id)[0]
		new_pos = np.array(pred)+np.array(link_pos)-env.tool_pos

		new_joint_positions = np.array(p.calculateInverseKinematics(env.robot, 13, new_pos, physicsClientId=env.id))
		new_joint_positions = new_joint_positions[:7]
		action = new_joint_positions - joint_positions

		clip_by_norm = lambda traj,limit: traj/max(1e-4,norm(traj))*np.clip(norm(traj),None,limit)
		action = clip_by_norm(action,self.action_clip)

		# p.removeBody(self.new_pos)
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[10, 255, 10, 1], physicsClientId=env.id)
		# self.new_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
		# 		basePosition=pred, useMaximalCoordinates=False, physicsClientId=env.id)

		# p.removeBody(self.tool_pos)
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=env.id)
		# self.tool_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
		# 		basePosition=env.tool_pos, useMaximalCoordinates=False, physicsClientId=env.id)

		return action

def sparse_factory(base):
	class Sparse(Env):
		def __init__(self,config):
			self.env = {
				"ScratchItch": assistive_gym.ScratchItchJacoEnv,
				"Feeding": assistive_gym.FeedingJacoEnv,
				"Laptop": assistive_gym.LaptopJacoEnv,
				"LightSwitch": assistive_gym.LightSwitchJacoEnv,
				"Reach": assistive_gym.ReachJacoEnv,
			}[config['env_name']](**config['env_kwargs'])

			self.step_limit = config['step_limit']

		def step(self,action):
			obs,r,done,info = self.env.step(action)
			self.timesteps += 1

			r = self.env.task_success > 0
			done = self.env.task_success > 0 or self.timesteps >= self.step_limit

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
			self.input_penalty = config['input_penalty']
			self.action_type = config['action_type']
			self.oracle = config['oracle'](self.env)
			self.current_obs_dim = np.prod(self.env.observation_space.shape)+self.oracle.size
			self.observation_space = spaces.Box(-np.inf,np.inf,(self.current_obs_dim,))

			if config['action_type'] in ['target', 'trajectory','disc_traj']:
				self.pretrain = IKAgent(config)
			joint = lambda action: action
			target = lambda pred: self.pretrain.predict(self.env,pred)
			trajectory = lambda traj: target(self.env.tool_pos+traj)
			def disc_traj(index):
				index = np.argmax(index)
				traj = [
					np.array((-1,0,0)),
					np.array((1,0,0)),
					np.array((0,-1,0)),
					np.array((0,1,0)),
					np.array((0,0,-1)),
					np.array((0,0,1)),
				][index]
				return trajectory(traj)
			self.translate = {
				# 'target': target,
				'trajectory': trajectory,
				'joint': joint,
				'disc_traj': disc_traj,
			}[config['action_type']]
			self.action_space = {
				# "target": spaces.Box(-1,1,(3,)),
				"trajectory": spaces.Box(-1,1,(3,)),
				"joint": spaces.Box(-1,1,(7,)),
				"disc_traj": spaces.Box(0,1,(6,)),
			}[config['action_type']]

		def step(self,action):
			t_action = self.translate(action)
			obs,r,done,info = super().step(t_action)
			obs = self.predict(obs,info)
			r -= self.input_penalty*(not info['noop'])

			if self.action_type in ['target']:
				info['pred_target_dist'] = norm(action-self.env.target_pos)

			return obs,r,done,info

		def reset(self):
			obs = super().reset()
			self.oracle.reset()
			obs = np.concatenate((obs,np.zeros(self.oracle.size)))
			return obs

		def predict(self,obs,info):
			recommend,_info = self.oracle.get_action(obs,info)

			info['recommend'] = recommend
			info['noop'] = not np.count_nonzero(recommend)

			return np.concatenate((obs,recommend))

	return SharedAutonomy

def window_factory(base):
	class PrevNnonNoopK(base):
		def __init__(self,config):
			super().__init__(config)
			self.current_obs_dim = np.prod(self.env.observation_space.shape)+self.oracle.size
			self.history_shape = (config['num_obs'],self.current_obs_dim)
			self.nonnoop_shape = (config['num_nonnoop'],self.current_obs_dim)
			# self.observation_space = spaces.Box(-np.inf,np.inf,(np.prod(self.history_shape)+np.prod(self.nonnoop_shape)+self.env.num_targets*3,))
			self.observation_space = spaces.Box(-np.inf,np.inf,(np.prod(self.history_shape)+np.prod(self.nonnoop_shape),))

		def step(self,action):
			obs,r,done,info = super().step(action)

			if len(self.history) == self.history_shape[0] and self.is_nonnoop[0]:
				self.prev_nonnoop.append(self.history[0])

			self.history.append(obs)
			self.is_nonnoop.append((not info['noop']))
			info['current_obs'] = obs

			# return np.concatenate((*self.env.targets,*self.prev_nonnoop,*self.history,)),r,done,info
			return np.concatenate((*self.prev_nonnoop,*self.history,)),r,done,info

		def reset(self):
			obs = super().reset()
			if obs is None:
				return self.observation_space.sample()
			self.history = deque(np.zeros(self.history_shape),self.history_shape[0])
			self.is_nonnoop = deque([False]*self.history_shape[0],self.history_shape[0])
			self.prev_nonnoop = deque(np.zeros(self.nonnoop_shape),self.nonnoop_shape[0])
			self.history.append(obs)

			# return np.concatenate((*self.env.targets,*self.prev_nonnoop,*self.history,))
			return np.concatenate((*self.prev_nonnoop,*self.history,))
	return PrevNnonNoopK

def cap_factory(base):
	class CapReward(base):
		def __init__(self,config):
			super().__init__(config)
			self.cap = config['cap']
		def step(self,action):
			obs,r,done,info = super().step(action)
			r = np.minimum(r,self.cap)
			return obs,r,done,info
	return CapReward

def target_factory(base):
	class Target(base):
		def __init__(self,config):
			super().__init__(config)
			self.observation_space = spaces.Box(-np.inf,np.inf,(np.prod(self.observation_space.shape)+3,))
		def step(self,action):
			obs,r,done,info = super().step(action)
			obs = np.concatenate((self.env.target_pos,obs,))
			return obs,r,done,info
		def reset(self):
			obs = super().reset()
			obs = np.concatenate((self.env.target_pos,obs,))
			return obs
	return Target

def metric_factory(base):
	class Metric(base):
		def __init__(self,config):
			super().__init__(config)
			self.success_count = deque([0]*20,20)
			self.success_dist = .2
		def step(self,action):
			obs,r,done,info = super().step(action)
			if done:
				self.success_count.append(info['task_success'])
			info['success_dist'] = self.success_dist
			return obs,r,done,info
		def reset(self):
			if np.mean(self.success_count) > .5:
				self.success_dist = max(.025,self.success_dist*.95)
				self.success_count = deque([0]*20,20)
			self.env.success_dist = self.success_dist
			obs = super().reset()
			return obs
	return Metric

# import pygame as pg
# def feedback_factory(base):
# 	SCREEN_SIZE = 250
# 	class Feedback(base):
# 		def __init__(self,config):
# 			super().__init__(config)
# 			self.screen = None

# 		def get_user_input(self,obs,info):
# 			# if self.screen is not None:
# 			# 	self.screen.fill(pg.color.THECOLORS["white"])
# 			# 	font = pg.font.Font(None, 24)
# 			# 	text = label
# 			# 	text = font.render(text, 1, pg.color.THECOLORS["black"])
# 			# 	self.screen.blit(text, (125,125))
# 			pass

# 		def render(self,mode=None):
# 			# if mode == 'human':
# 				# pg.init()
# 				# self.screen = pg.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
# 			return super().render(mode)
# 	return Feedback

default_config = {
	'step_limit': 200,
	'env_kwargs': {},
	'noise': False,
}
wrapper = lambda factories,base_factory: reduce(lambda value,func: func(value), factories, base_factory)
default_class = wrapper([cap_factory,shared_autonomy_factory],sparse_factory(None))