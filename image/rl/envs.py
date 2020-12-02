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
from railrl.envs.env_utils import get_dim

from agents import KeyboardOracle,MouseOracle,UserModelOracle,DemonstrationPolicy,TranslationPolicy

def default_overhead(config):
	if 'oracle' in config:
		config['factories'] = [oracle_factory] + config['factories']
	factories = [action_factory] + config['factories']
	wrapper = reduce(lambda value,func: func(value), factories, AssistiveWrapper)
	
	class Overhead(wrapper):
		def __init__(self,config):
			self.rng = default_rng(config['seedid'])
			super().__init__(config)
			adapt_map = {
				'burst': burst,
				'stack': stack,
				'reward': reward,
				'train': train_li_oracle,
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
		self.step_limit = config['step_limit']
		self.observation_space = self.base_env.observation_space
		self.action_space = self.base_env.action_space

	def step(self,action):
		obs,r,done,info = self.base_env.step(action)
		self.timesteps += 1

		if self.env_name in ['Circle','Sin']:
			info['frachet'] = self.base_env.discrete_frachet/self.timesteps
			info['task_success'] = self.timesteps >= self.step_limit and info['fraction_t'] >= .8

		done = self.base_env.task_success > 0 or self.timesteps >= self.step_limit
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

		def step(self,action):
			obs,r,done,info = super().step(action)
			if self.action_type in ['target']:
				info['pred_target_dist'] = norm(action-self.base_env.target_pos)
			return obs,r,done,info
	return Action

def oracle_factory(base):
	class Oracle(base):
		def __init__(self,config):
			super().__init__(config)
			self.input_penalty = config['input_penalty']
			self.oracle = {
				'keyboard': KeyboardOracle,
				'mouse': MouseOracle,
				'model': UserModelOracle,
			}[config['oracle']](self,**config['oracle_kwargs'])
			self.observation_space = spaces.Box(-10,10,
									(get_dim(self.observation_space)+self.oracle.size,))

		def step(self,action):
			obs,r,done,info = super().step(action)
			obs = self._predict(obs,info)
			return obs,r,done,info

		def reset(self):
			obs = super().reset()
			self.oracle.reset()
			self.recommend = np.zeros(6)
			return np.concatenate((obs,np.zeros(6)))
			# return obs

		def _predict(self,obs,info):
			recommend,_info = self.oracle.get_action(obs,info)
			self.recommend = recommend
			# info['recommend'] = recommend
			info['noop'] = not np.count_nonzero(recommend)
			# return obs
			return np.concatenate((obs,recommend))
	return Oracle

def train_oracle_factory(base):
	class TrainOracle(base):
		def __init__(self,config):
			super().__init__(config)
			# self.initial_oracle = UserModelOracle(self,**config['oracle_kwargs'])
			# self.initial_policy = TranslationPolicy(self,DemonstrationPolicy(self,lower_p=.8),**config)
			self.observation_space = spaces.Box(-10,10,
									(get_dim(self.observation_space)+3+3+3*3+3,))
		def step(self,action):
			obs,r,done,info = super().step(action)
			base_env = self.base_env
			bad_switch = np.logical_and(info['angle_dir'] != 0,
										np.logical_or(np.logical_and(info['angle_dir'] < 0, base_env.target_string == 1),
											np.logical_and(info['angle_dir'] > 0, base_env.target_string == 0))).astype(int)
			bad_contact = (info['bad_contact'] > 0)\
						or (np.count_nonzero(bad_switch)>0)
			info['bad_contact'] = bad_contact
			switch_pos,__ = p.getBasePositionAndOrientation(base_env.switches[0], physicsClientId=base_env.id)
			obs = np.concatenate([base_env.target_string,base_env.current_string,*base_env.target_pos,switch_pos,obs]).ravel()
			return obs,r,done,info
		def reset(self):
			# if self.rng.random() < 1/3:
			# 	def init_start_pos(self,og_init_pos):
			# 		switch_pos, switch_orient = p.getBasePositionAndOrientation(self.switches[1], physicsClientId=self.id)
			# 		init_pos, __ = p.multiplyTransforms(switch_pos, switch_orient, [0,.3,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
			# 		return init_pos
			# 	self.base_env.init_start_pos = MethodType(init_start_pos,self.base_env)
			# 	obs = super().reset()
			# 	angle = p.getJointStates(self.base_env.switches[0], jointIndices=[0], physicsClientId=self.base_env.id)[0][0]
			# 	p.resetJointState(self.base_env.switches[0], jointIndex=0, targetValue=-1-angle, physicsClientId=self.base_env.id)
			# elif self.rng.random() < 1/2:
			# 	def init_start_pos(self,og_init_pos):
			# 		switch_pos, switch_orient = p.getBasePositionAndOrientation(self.switches[2], physicsClientId=self.id)
			# 		init_pos, __ = p.multiplyTransforms(switch_pos, switch_orient, [0,.3,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
			# 		return init_pos
			# 	self.base_env.init_start_pos = MethodType(init_start_pos,self.base_env)
			# 	obs = super().reset()
			# 	angle = p.getJointStates(self.base_env.switches[0], jointIndices=[0], physicsClientId=self.base_env.id)[0][0]
			# 	p.resetJointState(self.base_env.switches[0], jointIndex=0, targetValue=-1-angle, physicsClientId=self.base_env.id)
			# 	angle = p.getJointStates(self.base_env.switches[1], jointIndices=[0], physicsClientId=self.base_env.id)[0][0]
			# 	p.resetJointState(self.base_env.switches[1], jointIndex=0, targetValue=-1-angle, physicsClientId=self.base_env.id)
			# else:
			# 	obs = super().reset()

			# self.initial_oracle.reset()
			# self.initial_policy.reset()
			# obs,r,done,info = self.step(np.zeros(7))
			# bad_contact_found = False
			# for i in range(100):
			# 	self.recommend,_info = self.initial_oracle.get_action(obs,info)
			# 	if info['bad_contact']:
			# 		bad_contact_found = True
			# 		break
			# 	action,_info = self.initial_policy.get_action(obs)
			# 	obs,r,done,info = self.step(action)
			# 	self.timestep = 0
			# if not bad_contact_found:
			# 	return self.reset()
			# return obs

			obs = super().reset()
			base_env = self.base_env
			switch_pos,__ = p.getBasePositionAndOrientation(base_env.switches[0], physicsClientId=base_env.id)
			return np.concatenate([base_env.target_string,base_env.current_string,*base_env.target_pos,switch_pos,obs]).ravel()
	return TrainOracle

class train_li_oracle:
	def __init__(self,master_env,config):
		pass
	def _step(self,obs,r,done,info):
		r = 0
		target_string = obs[:3]
		current_string = obs[3:6]
		target_pos = obs[6:15].reshape((3,3))
		tool_pos = obs[-15:-12]
		target_indices = np.nonzero(np.not_equal(target_string,current_string))[0]
		if len(target_indices) > 0:
			# r -= min([norm(self.tool_pos-self.target_pos[i]) for i in target_indices])
			r -= 10*norm(tool_pos-target_pos[target_indices[0]])
		else:
			r -= 0
		for i in range(3):
			if target_string[i] == 0:
				r -= 250*abs(-.02 - info['angle_dir'][i])
			else:
				r -= 250*abs(.02 - info['angle_dir'][i])
		r -= 10*len(target_indices)
		print(obs[:6],info['angle_dir'],r)
		return obs,r,done,info
	def _reset(self,obs):
		return obs
		
class burst:
	""" Remove user input from observation and reward if input is in bursts
		Burst defined as inputs separated by no more than 'space' steps """
	def __init__(self,master_env,config):
		self.space = config['space']+2

	def _step(self,obs,r,done,info):
		if np.count_nonzero(obs[-6:]):
			if self.timer < self.space:
				obs[-6:] = np.zeros(6)
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
		self.master_env = master_env

	def _step(self,obs,r,done,info):
		r = 0
		# r += info['task_success']
		r -= self.master_env.input_penalty*(np.count_nonzero(obs[-6:]) > 0)
		r = np.clip(r,*self.range)
		return obs,r,done,info

	def _reset(self,obs):
		return obs

import pygame as pg
SCREEN_SIZE = 300
def render_user_factory(base):
	class RenderUser(base):
		def __init__(self,config):
			super().__init__(config)

		def setup_render(self):
			pg.init()
			self.screen = pg.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))
			pg.mouse.set_visible(False)

		def render_input(self):
			self.screen.fill(pg.color.THECOLORS["white"])

			center = np.array([SCREEN_SIZE//2,SCREEN_SIZE//2])
			mouse_pos = self.oracle_info['mouse_pos']
			choice_action = self.oracle_info['action']
			rad2p = lambda rad: np.array([np.cos(rad),np.sin(rad)])
			for i in range(1,13,2):
				pg.draw.line(self.screen, (10,10,10,0), 1000*rad2p(np.pi*i/6)+center, 50*rad2p(np.pi*i/6)+center, 2)
			pg.draw.circle(self.screen, (10,10,10,0), center, 50, 2)

			font = pg.font.Font(None, 24)
			for action,i in zip(['right','left','forward','backward','up','down'],[0,3,5,2,4,1]):
				if action == choice_action:
					text = font.render(action, 1, pg.color.THECOLORS["blue"])
				else:
					text = font.render(action, 1, pg.color.THECOLORS["black"])
				self.screen.blit(text, 80*rad2p(np.pi*i/3)+center-np.array([text.get_width()/2,text.get_height()/2]))
			if 'noop' == choice_action:
				text = font.render('noop', 1, pg.color.THECOLORS["blue"])
			else:
				text = font.render('noop', 1, pg.color.THECOLORS["black"])
			self.screen.blit(text, center-np.array([text.get_width()/2,text.get_height()/2]))

			pg.draw.circle(self.screen, (76,187,23,0), mouse_pos, 5)

			pg.display.flip()
			pg.event.pump()

		def step(self,action):
			obs,r,done,info = super().step(action)
			self.render_input()
			return obs,r,done,info

		def reset(self):
			if self.setup_render:
				self.setup_render()
				self.setup_render = None
			obs = super().reset()
			return obs
	return RenderUser
