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
from gym import spaces, Env
from rlkit.envs.env_utils import get_dim
import gym

from rl.oracles import *


def default_overhead(config):
	factories = [action_factory]
	wrapper = reduce(lambda value, func: func(value), factories, LibraryWrapper)

	class Overhead(wrapper):
		def __init__(self, config):
			self.rng = default_rng(config['seedid'])
			super().__init__(config)
			adapt_map = {
				'burst': burst,
				'high_dim_user': high_dim_user,
				'stack': stack,
				'reward': reward,
			}
			self.adapts = [adapt_map[adapt] for adapt in config['adapts']]
			self.adapts = [oracle] + self.adapts
			self.adapts = [adapt(self, config) for adapt in self.adapts]
			self.adapt_step = lambda obs, r, done, info: reduce(lambda sub_tran, adapt: adapt._step(*sub_tran),
																self.adapts, (obs, r, done, info))
			self.adapt_reset = lambda obs: reduce(lambda obs, adapt: adapt._reset(obs), self.adapts, (obs))

		def step(self, action):
			tran = super().step(action)
			tran = self.adapt_step(*tran)
			return tran

		def reset(self):
			obs = super().reset()
			obs = self.adapt_reset(obs)
			return obs

	return Overhead(config)


class LibraryWrapper(Env):
	def __init__(self, config):
		self.env_name = config['env_name']

		self.base_env, self.env_name = {
			"Feeding": (ag.FeedingJacoEnv, 'Feeding'),
			"Laptop": (ag.LaptopJacoEnv, 'Laptop'),
			"OneSwitch": (ag.OneSwitchJacoEnv, 'OneSwitch'),
			"ThreeSwitch": (ag.ThreeSwitchJacoEnv, 'ThreeSwitch'),
			"Bottle": (ag.BottleJacoEnv, 'Bottle'),
			"Kitchen": (ag.KitchenJacoEnv, 'Kitchen'),
			# "Circle": (ag.CircleJacoEnv, 'Circle'),
			# "Sin": (ag.SinJacoEnv, 'Sin'),
		}[config['env_name']]
		self.base_env = self.base_env(**config['env_kwargs'])
		self.observation_space = self.base_env.observation_space
		self.action_space = self.base_env.action_space
		self.step_limit = config['step_limit']

	def step(self, action):
		obs, r, done, info = self.base_env.step(action)
		info['raw_obs'] = obs

		if self.env_name in ['Circle', 'Sin']:
			self.timesteps += 1
			info['frachet'] = self.base_env.discrete_frachet / self.timesteps
			info['task_success'] = self.timesteps >= self.step_limit and info['fraction_t'] >= .8

		done = info['task_success']
		# info['target_pos'] = self.base_env.target_pos
		return obs, r, done, info

	def reset(self):
		obs = self.base_env.reset()
		self.timesteps = 0
		return obs

	def render(self, mode=None, **kwargs):
		return self.base_env.render(mode)

	def seed(self, value):
		self.base_env.seed(value)

	def close(self):
		self.base_env.close()

	def get_base_env(self):
		return self.base_env


def action_factory(base):
	class Action(base):
		def __init__(self, config):
			super().__init__(config)
			self.action_type = config['action_type']
			self.action_space = {
				"trajectory": spaces.Box(-.1, .1, (3,)),
				"joint": spaces.Box(-.25, .25, (7,)),
				"disc_traj": spaces.Box(0, 1, (6,)),
			}[config['action_type']]
			self.translate = {
				# 'target': target,
				'trajectory': self.trajectory,
				'joint': self.joint,
				'disc_traj': self.disc_traj,
			}[config['action_type']]
			self.smooth_alpha = config['smooth_alpha']

		def joint(self, action, info={}):
			clip_by_norm = lambda traj, limit: traj / max(1e-4, norm(traj)) * np.clip(norm(traj), None, limit)
			action = clip_by_norm(action, .25)
			self.action = self.smooth_alpha * action + (1 - self.smooth_alpha) * self.action if np.count_nonzero(
				self.action) else action
			info['joint'] = self.action
			return action, info

		def target(self, coor, info={}):
			base_env = self.base_env
			info['target'] = coor
			joint_states = p.getJointStates(base_env.robot, jointIndices=base_env.robot_left_arm_joint_indices,
											physicsClientId=base_env.id)
			joint_positions = np.array([x[0] for x in joint_states])

			link_pos = p.getLinkState(base_env.robot, 13, computeForwardKinematics=True, physicsClientId=base_env.id)[0]
			new_pos = np.array(coor) + np.array(link_pos) - base_env.tool_pos

			new_joint_positions = np.array(
				p.calculateInverseKinematics(base_env.robot, 13, new_pos, physicsClientId=base_env.id))
			new_joint_positions = new_joint_positions[:7]
			action = new_joint_positions - joint_positions
			return self.joint(action, info)

		def trajectory(self, traj, info={}):
			clip_by_norm = lambda traj, min_l=None, max_l=None: traj / max(1e-4, norm(traj)) * np.clip(norm(traj),
																									   min_l, max_l)
			traj = clip_by_norm(traj, .07, .1)
			info['trajectory'] = traj
			return self.target(self.base_env.tool_pos + traj, info)

		def disc_traj(self, onehot, info={}):
			info['disc_traj'] = onehot
			index = np.argmax(onehot)
			traj = [
				np.array((-1, 0, 0)),
				np.array((1, 0, 0)),
				np.array((0, -1, 0)),
				np.array((0, 1, 0)),
				np.array((0, 0, -1)),
				np.array((0, 0, 1)),
			][index]
			return self.trajectory(traj, info)

		def step(self, action):
			action, ainfo = self.translate(action)
			obs, r, done, info = super().step(action)
			info.update(ainfo)
			if self.action_type in ['target']:
				info['pred_target_dist'] = norm(action - self.base_env.target_pos)
			return obs, r, done, info

		def reset(self):
			self.action = np.zeros(7)
			return super().reset()

	return Action

class oracle:
	def __init__(self,master_env,config):
		self.oracle_type = config['oracle']
		self.input_in_obs = config.get('input_in_obs',False)
		if self.oracle_type == 'model':
			self.oracle = master_env.oracle = {
				"Feeding": StraightLineOracle,
				"Laptop": LaptopOracle,
				"OneSwitch": OneSwitchOracle,
				"ThreeSwitch": ThreeSwitchOracle,
				"Bottle": BottleOracle,
				"Kitchen": StraightLineOracle,

				# "Circle": TracingOracle,
				# "Sin": TracingOracle,
			}[master_env.env_name](master_env.rng,**config['oracle_kwargs'])
		else:
			self.oracle = master_env.oracle = {
				'keyboard': KeyboardOracle,
				# 'mouse': MouseOracle,
			}[config['oracle']](master_env)
		self.full_obs_size = get_dim(master_env.observation_space)+self.oracle.size
		if self.input_in_obs:
			master_env.observation_space = spaces.Box(-np.inf,np.inf, (self.full_obs_size,))
		self.master_env = master_env

	def _step(self,obs,r,done,info):
		# obs = info['raw_obs']
		if not self.input_in_obs and obs.size > self.full_obs_size-self.oracle.size: # only true if trans from demo
			obs = obs[-(self.full_obs_size-self.oracle.size):]
		if obs.size < self.full_obs_size and self.input_in_obs: # not 'model' case is depricated in this code
			obs = self._predict(obs,info)
		else:
			self._predict(obs,info)
		return obs,r,done,info

	def _reset(self,obs):
		self.oracle.reset()
		self.master_env.recommend = np.zeros(self.oracle.size)
		if not self.input_in_obs and obs.size > self.full_obs_size-self.oracle.size:
			obs = obs[-(self.full_obs_size-self.oracle.size):]
		elif obs.size < self.full_obs_size and self.input_in_obs:
			obs = np.concatenate((obs,self.master_env.recommend))
		return obs

	def _predict(self,obs,info):
		recommend,_info = self.oracle.get_action(obs,info)
		self.master_env.recommend = info['recommend'] = recommend
		info['noop'] = not np.count_nonzero(recommend)
		return np.concatenate((obs,recommend))

class high_dim_user:
	def __init__(self,master_env,config):
		master_env.observation_space = spaces.Box(-np.inf,np.inf,(get_dim(master_env.observation_space)+50,))
		self.env_name = master_env.env_name
		self.random_projection = master_env.rng.normal(0,10,(50,50))
		self.random_projection,*_ = np.linalg.qr(self.random_projection)
		self.apply_projection = config['apply_projection']
		
		self.state_type = config['state_type']


	def _step(self,obs,r,done,info):
		state_func = {
			'OneSwitch': lambda: np.concatenate([np.ravel(info[state_component]) for state_component in [
					['switch_pos','tool_pos',],
					['aux_switch_pos','tool_pos',],
					['lever_angle','switch_pos','tool_pos',],
					['lever_angle','target_string','current_string',
					'switch_pos','aux_switch_pos','tool_pos',],
					][self.state_type]]),
			'ThreeSwitch': lambda: np.concatenate([np.ravel(info[state_component]) for state_component in 
					['lever_angle','target_string','current_string',
					 'switch_pos','aux_switch_pos','tool_pos',]]),
			'Laptop': lambda: np.concatenate([np.ravel(info[state_component]) for state_component in 
					['target_pos','lid_pos','tool_pos','lever_angle',]]),
			'Bottle': lambda: np.concatenate([np.ravel(info[state_component]) for state_component in [
						['bottle_pos','tool_pos',],
						['target1_reached','bottle_pos','tool_pos'],
						['target1_pos','bottle_pos','tool_pos',],
						['target_pos','target1_pos','bottle_pos','tool_pos',],
						['aux_target_pos','target_pos','target1_pos','bottle_pos','tool_pos',],
						# ['cos_error','aux_target_pos','target_pos','target1_pos','bottle_pos','tool_pos',],
					][self.state_type]]),
		}[self.env_name]()
		state_func = np.concatenate((state_func,np.zeros(50-state_func.size)))
		if self.apply_projection:
			state_func = state_func @ self.random_projection
		obs = np.concatenate((state_func,obs))
		return obs,r,done,info

	def _reset(self,obs):
		return np.concatenate((np.zeros(50),obs,))

class burst:
	""" Remove user input from observation and reward if input is in bursts
		Burst defined as inputs separated by no more than 'space' steps """

	def __init__(self, master_env, config):
		self.space = config['space'] + 2
		self.master_env = master_env

	def _step(self, obs, r, done, info):
		oracle_size = self.master_env.oracle.size
		if np.count_nonzero(obs[-oracle_size:]):
			if self.timer < self.space:
				obs[-oracle_size:] = np.zeros(6)
			self.timer = 0
		self.timer += 1
		return obs, r, done, info

	def _reset(self, obs):
		self.timer = self.space
		obs, _r, _d, _i = self._step(obs, 0, False, {})
		return obs


class stack:
	""" 'num_obs' most recent steps and 'num_nonnoop' most recent input steps stacked """

	def __init__(self, master_env, config):
		self.history_shape = (config['num_obs'], get_dim(master_env.observation_space))
		self.nonnoop_shape = (config['num_nonnoop'], get_dim(master_env.observation_space))
		master_env.observation_space = spaces.Box(-np.inf, np.inf,
												  (np.prod(self.history_shape) + np.prod(self.nonnoop_shape),))
		self.master_env = master_env

	def _step(self, obs, r, done, info):
		if len(self.history) == self.history.maxlen:
			old_obs = self.history[0]
			oracle_size = self.master_env.oracle.size
			if np.count_nonzero(old_obs[-oracle_size:]) > 0:
				self.prev_nonnoop.append(old_obs)
		self.history.append(obs)
		# info['current_obs'] = obs
		return np.concatenate((*self.prev_nonnoop, *self.history,)), r, done, info

	def _reset(self, obs):
		self.history = deque(np.zeros(self.history_shape), self.history_shape[0])
		self.prev_nonnoop = deque(np.zeros(self.nonnoop_shape), self.nonnoop_shape[0])
		obs, _r, _d, _i = self._step(obs, 0, False, {})
		return obs


class reward:
	""" rewards capped at 'cap' """

	def __init__(self, master_env, config):
		self.range = (config['reward_min'], config['reward_max'])
		self.input_penalty = config['input_penalty']
		self.master_env = master_env
		self.reward_type = config['reward_type']

	def _step(self, obs, r, done, info):
		if self.reward_type == 'user_penalty':
			r = 0
			r -= self.input_penalty * (not info['noop'])
			# if info['task_success']:
			#     r = 1
			done = info['task_success']
		elif self.reward_type == 'custom':
			r = 0
			if not info['task_success']:
				target_indices = np.nonzero(np.not_equal(info['target_string'], info['current_string']))[0]
				target_pos = np.array(info['switch_pos'])[target_indices[0]]
				r += -1 + (norm(info['old_tool_pos'] - target_pos) - norm(info['tool_pos'] - target_pos))
				# done = done
		elif self.reward_type == 'sparse':
			r = -1 + info['task_success']
			done = info['task_success']
		elif self.reward_type == 'part_sparse':
			r = -1 + .5*(info['task_success']+info['target1_reached'])
			done = info['task_success']
		elif self.reward_type == 'terminal_interrupt':
			r = info['noop']
			# done = info['noop']
			done = info['task_success']
		else:
			error

		r = np.clip(r, *self.range)

		return obs, r, done, info

	def _reset(self, obs):
		return obs
