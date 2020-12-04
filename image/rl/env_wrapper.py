from functools import reduce
from collections import deque
import os
from copy import deepcopy
from types import MethodType

import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm

import pybullet as p
import assistive_gym as ag
from gym import spaces,Env
from rlkit.envs.env_utils import get_dim

from .oracles import *

def default_overhead(config):
	factories = [oracle_factory,action_factory]
	wrapper = reduce(lambda value,func: func(value), factories, AssistiveWrapper)
	
	class Overhead(wrapper):
		def __init__(self,config):
			self.rng = default_rng(config['seedid'])
			super().__init__(config)
			adapt_map = {
				'burst': burst,
				'stack': stack,
				'reward': reward,
			}
			self.adapts = [adapt_map[adapt] for adapt in config['adapts']]
			self.adapts = [adapt(self,config) for adapt in self.adapts]
			self.adapt_step = lambda obs,r,done,info:reduce(lambda sub_tran,adapt:adapt._step(*sub_tran),
														self.adapts,(obs,r,done,info))
			self.adapt_reset = lambda obs:reduce(lambda obs,adapt:adapt._reset(obs),self.adapts,(obs))

		def step(self,action):
			tran = super().step(action)
			tran = self.adapt_step(*tran)
			return tran
		
		def reset(self):
			obs = super().reset()
			obs = self.adapt_reset(obs)
			return obs
	return Overhead(config)

class AssistiveWrapper(Env):
	def __init__(self,config):
		self.env_name = config['env_name']
		
		self.base_env,self.env_name = {
			"Feeding": (ag.FeedingJacoEnv, 'Feeding'),
			"Laptop": (ag.LaptopJacoEnv, 'Laptop'),
			"LightSwitch": (ag.LightSwitchJacoEnv, 'LightSwitch'),
			"Circle": (ag.CircleJacoEnv, 'Circle'),
			"Sin": (ag.SinJacoEnv, 'Sin'),
		}[config['env_name']]
		self.base_env = self.base_env(**config['env_kwargs'])
		self.observation_space = self.base_env.observation_space
		self.action_space = self.base_env.action_space
		self.step_limit = config['step_limit']

	def step(self,action):
		obs,r,done,info = self.base_env.step(action)
		
		if self.env_name in ['Circle','Sin']:
			self.timesteps += 1
			info['frachet'] = self.base_env.discrete_frachet/self.timesteps
			info['task_success'] = self.timesteps >= self.step_limit and info['fraction_t'] >= .8

		# done = info['task_success']
		self.timesteps += 1
		done = info['task_success'] or self.timesteps >= self.step_limit
		info['target_pos'] = self.base_env.target_pos
		return obs,r,done,info

	def reset(self):
		obs = self.base_env.reset()
		self.timesteps = 0
		return obs

	def render(self,mode=None,**kwargs):
		return self.base_env.render(mode)
	def seed(self,value):
		self.base_env.seed(value)
	def close(self):
		self.base_env.close()

	def get_base_env(self):
		return self.base_env

def action_factory(base):
	class Action(base):
		def __init__(self,config):
			super().__init__(config)
			self.action_type = config['action_type']
			self.action_space = {
				"trajectory": spaces.Box(-1,1,(3,)),
				"joint": spaces.Box(-1,1,(7,)),
				"disc_traj": spaces.Box(0,1,(6,)),
			}[config['action_type']]
			self.translate = {
				# 'target': target,
				'trajectory': self.trajectory,
				'joint': self.joint,
				'disc_traj': self.disc_traj,
			}[config['action_type']]
			self.smooth_alpha = config['smooth_alpha']

		def joint(self,action,info={}):
			self.action = self.smooth_alpha*action + (1-self.smooth_alpha)*self.action if np.count_nonzero(self.action) else action
			info['joint'] = self.action
			return action,info	
		def target(self,coor,info={}):
			base_env = self.base_env
			info['target'] = coor
			joint_states = p.getJointStates(base_env.robot, jointIndices=base_env.robot_left_arm_joint_indices, physicsClientId=base_env.id)
			joint_positions = np.array([x[0] for x in joint_states])

			link_pos = p.getLinkState(base_env.robot, 13, computeForwardKinematics=True, physicsClientId=base_env.id)[0]
			new_pos = np.array(coor)+np.array(link_pos)-base_env.tool_pos

			new_joint_positions = np.array(p.calculateInverseKinematics(base_env.robot, 13, new_pos, physicsClientId=base_env.id))
			new_joint_positions = new_joint_positions[:7]
			action = new_joint_positions - joint_positions

			clip_by_norm = lambda traj,limit: traj/max(1e-4,norm(traj))*np.clip(norm(traj),None,limit)
			action = clip_by_norm(action,.25)
			return self.joint(action, info)
		def trajectory(self,traj,info={}):
			info['trajectory'] = traj
			return self.target(self.base_env.tool_pos+traj,info)
		def disc_traj(self,onehot,info={}):
			info['disc_traj'] = onehot
			index = np.argmax(onehot)
			traj = [
				np.array((-1,0,0)),
				np.array((1,0,0)),
				np.array((0,-1,0)),
				np.array((0,1,0)),
				np.array((0,0,-1)),
				np.array((0,0,1)),
			][index]*.1
			return self.trajectory(traj,info)
		
		def step(self,action):
			action,ainfo = self.translate(action)
			obs,r,done,info = super().step(action)
			info.update(ainfo)
			if self.action_type in ['target']:
				info['pred_target_dist'] = norm(action-self.base_env.target_pos)
			return obs,r,done,info

		def reset(self):
			self.action = np.zeros(7)
			return super().reset()

	return Action

def oracle_factory(base):
	class Oracle(base):
		def __init__(self,config):
			super().__init__(config)
			if config['oracle'] == 'model':
				self.oracle = {
					"Feeding": StraightLineOracle,
					"Laptop": StraightLineOracle,
					"LightSwitch": LightSwitchOracle,
					"Circle": TracingOracle,
					"Sin": TracingOracle,
				}[self.env_name](self.rng,self.base_env,**config['oracle_kwargs'])
			else:
				self.oracle = {
					'keyboard': KeyboardOracle,
					# 'mouse': MouseOracle,
				}[config['oracle']](self)
			self.observation_space = spaces.Box(-10,10,
									(get_dim(self.observation_space)+self.oracle.size,))

		def step(self,action):
			obs,r,done,info = super().step(action)
			obs = self._predict(obs,info)
			return obs,r,done,info

		def reset(self):
			obs = super().reset()
			self.oracle.reset()
			return np.concatenate((obs,np.zeros(self.oracle.size)))

		def _predict(self,obs,info):
			recommend,_info = self.oracle.get_action(obs,info)
			info['noop'] = not np.count_nonzero(recommend)
			return np.concatenate((obs,recommend))
	return Oracle

class burst:
	""" Remove user input from observation and reward if input is in bursts
		Burst defined as inputs separated by no more than 'space' steps """
	def __init__(self,master_env,config):
		self.space = config['space']+2
		self.master_env = master_env

	def _step(self,obs,r,done,info):
		oracle_size = self.master_env.oracle.size
		if np.count_nonzero(obs[-oracle_size:]):
			if self.timer < self.space:
				obs[-oracle_size:] = np.zeros(6)
			self.timer = 0
		self.timer += 1
		return obs,r,done,info

	def _reset(self,obs):
		self.timer = self.space
		obs,_r,_d,_i = self._step(obs,0,False,{})
		return obs

class stack:
	""" 'num_obs' most recent steps and 'num_nonnoop' most recent input steps stacked """
	def __init__(self,master_env,config):
		self.history_shape = (config['num_obs'],get_dim(master_env.observation_space))
		self.nonnoop_shape = (config['num_nonnoop'],get_dim(master_env.observation_space))
		master_env.observation_space = spaces.Box(-np.inf,np.inf,(np.prod(self.history_shape)+np.prod(self.nonnoop_shape),))

	def _step(self,obs,r,done,info):
		self.history.append(obs)
		if not info.get('noop',True):
			self.prev_nonnoop.append(obs)
		# info['current_obs'] = obs
		return np.concatenate((*self.prev_nonnoop,*self.history,)),r,done,info

	def _reset(self,obs):
		self.history = deque(np.zeros(self.history_shape),self.history_shape[0])
		self.prev_nonnoop = deque(np.zeros(self.nonnoop_shape),self.nonnoop_shape[0])
		obs,_r,_d,_i = self._step(obs,0,False,{})
		return obs

class reward:
	""" rewards capped at 'cap' """
	def __init__(self,master_env,config):
		self.range = (config['reward_min'],config['reward_max'])
		self.input_penalty = config['input_penalty']
		self.master_env = master_env

	def _step(self,obs,r,done,info):
		r = 0
		# oracle_size = self.master_env.oracle.size
		# r -= self.input_penalty*(np.count_nonzero(obs[-oracle_size:]) > 0)
		# r = np.clip(r,*self.range)
		# done = info['task_success']

		r += info['task_success']
		return obs,r,done,info

	def _reset(self,obs):
		return obs
