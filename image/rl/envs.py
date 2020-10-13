from functools import reduce
from collections import deque
import os

import numpy as np
from numpy.random import default_rng
rng = default_rng()
from numpy.linalg import norm

import pybullet as p
import assistive_gym
from gym import spaces,Env
from railrl.envs.env_utils import get_dim

from agents import KeyboardOracle,MouseOracle,UserModelOracle

def overhead_factory(config):
	factories = [oracle_factory,action_factory,]+config['factories']
	wrapper = reduce(lambda value,func: func(value), factories, AssistiveWrapper)
	
	class Overhead(wrapper):
		def __init__(self,config):
			super().__init__(config)
			self.adapt_tran = config['adapt_tran']
			if self.adapt_tran:
				self.adapts = [adapt(config) for adapt in [burst_adapt,stack_adapt,cap_adapt]+config.get('adapts',[])]
				self.adapt_step = lambda obs,r,done,info:reduce(lambda sub_tran,adapt:adapt._step(*sub_tran),
															self.adapts,(obs,r,done,info))
				self.adapt_reset = lambda obs:reduce(lambda obs,adapt:adapt._step(obs),self.adapts,(obs))

		def step(self,action):
			tran = super().step(action)
			if self.adapt_tran:
				tran = self.adapt_step(*tran)
			return *tran
		
		def reset(self):
			obs = super().reset()
			if self.adapt_tran:
				obs = self.adapt_reset(obs)
			return obs
	return Overhead(config)

class AssistiveWrapper(Env):
	def __init__(self,config):
		self.env = {
			"ScratchItch": assistive_gym.ScratchItchJacoEnv,
			"Feeding": assistive_gym.FeedingJacoEnv,
			"Laptop": assistive_gym.LaptopJacoEnv,
			"LightSwitch": assistive_gym.LightSwitchJacoEnv,
			"Reach": assistive_gym.ReachJacoEnv,
		}[config['env_name']](**config['env_kwargs'])
		self.step_limit = config['step_limit']
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space

	def step(self,action):
		obs,r,done,info = self.env.step(action)
		self.timesteps += 1
		r = self.env.task_success > 0
		done = self.env.task_success > 0 or self.timesteps >= self.step_limit
		info['target_pos'] = self.env.target_index
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

def oracle_factory(base):
	class Oracle(base):
		def __init__(self,config):
			super().__init__(config)
			self.input_penalty = config['input_penalty']
			self.oracle = {
				'keyboard': KeyboardOracle,
				'mouse': MouseOracle,
				'usermodel': UserModelOracle,
			}[config['oracle'](self.env)]
			self.observation_space = spaces.Box(-10,10,
									(get_dim(self.observation_space)+self.oracle.size,))

		def step(self,action):
			obs,r,done,info = super().step(action)
			obs = self.predict(obs,info)
			r -= self.input_penalty*(not info['noop'])
			return obs,r,done,info

		def reset(self):
			obs = super().reset()
			self.oracle.reset()
			obs = np.concatenate((obs,np.zeros(self.oracle.size)))
			return obs

		def predict(self,obs,info):
			recommend,self.oracle_info = self.oracle.get_action(obs,info)
			# info['recommend'] = recommend
			info['noop'] = not np.count_nonzero(recommend)
			return np.concatenate((obs,recommend))

	return Oracle

def action_factory(base):
	class Action(base):
		def __init__(self,config):
			super().__init__(config)
			self.action_type = config['action_type']
			self.action_space = {
				# "target": spaces.Box(-1,1,(3,)),
				"trajectory": spaces.Box(-1,1,(3,)),
				"joint": spaces.Box(-1,1,(7,)),
				"disc_traj": spaces.Box(0,1,(6,)),
			}[config['action_type']]

		def step(self,action):
			t_action = self.translate(action)
			obs,r,done,info = super().step(t_action)
			if self.action_type in ['target']:
				info['pred_target_dist'] = norm(action-self.env.target_pos)
			return obs,r,done,info

	return Action

class burst_adapt:
	""" Remove user input from observation and reward if input is in bursts
		Burst defined as inputs separated by no more than 'space' steps """
	def __init__(self,config):
		self.space = config['space']+2

	def _step(self,obs,r,done,info):
		if np.count_nonzero(obs[-6:]):
			if timer < self.space:
				r = 0
				obs[-6:] = np.zeros(6)
			self.timer = 0
		self.timer += 1
		return obs,r,done,info

	def _reset(self,obs):
		self.timer = self.space
		obs,_r,_d,_i = self._step(obs,0,False,{})
		return obs

class stack_adapt:
	""" 'num_obs' most recent steps and 'num_nonnoop' most recent input steps stacked """
	def __init__(self,config):
		self.history_shape = (config['num_obs'],get_dim(self.observation_space))
		self.nonnoop_shape = (config['num_nonnoop'],get_dim(self.observation_space))
		self.observation_space = spaces.Box(-np.inf,np.inf,(np.prod(self.history_shape)+np.prod(self.nonnoop_shape),))

	def _step(self,obs,r,done,info):
		self.history.append(obs)
		if not info['noop']:
			self.prev_nonnoop.append(obs)
		# info['current_obs'] = obs
		return np.concatenate((*self.prev_nonnoop,*self.history,)),r,done,info

	def _reset(self,obs):
		self.history = deque(np.zeros(self.history_shape),self.history_shape[0])
		self.prev_nonnoop = deque(np.zeros(self.nonnoop_shape),self.nonnoop_shape[0])
		obs,_r,_d,_i = self._step(obs,0,False,{})
		return obs

class cap_adapt:
	""" rewards capped at 'cap' """
	def __init__(self,config):
		self.cap = config['cap']

	def _step(self,obs,r,done,info):
		r = np.minimum(r,self.cap)
		return obs,r,done,info

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
