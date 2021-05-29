from functools import reduce
from collections import deque
import os
from pathlib import Path
import h5py
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
	factory_map = {
				'session': session_factory,
			}
	factories = [factory_map[factory] for factory in config['factories']]
	factories = [action_factory]+factories
	wrapper = reduce(lambda value, func: func(value), factories, LibraryWrapper)

	class Overhead(wrapper):
		def __init__(self, config):
			self.rng = default_rng(config['seedid'])
			super().__init__(config)
			adapt_map = {
				'oracle': oracle,
				'static_gaze': static_gaze,
				'goal': goal,
				'reward': reward,
			}
			self.adapts = [adapt_map[adapt] for adapt in config['adapts']]
			# self.adapts = [array_to_dict] + self.adapts
			self.adapts = [adapt(self, config) for adapt in self.adapts]
			self.adapt_step = lambda obs, r, done, info: reduce(lambda sub_tran, adapt: adapt._step(*sub_tran),
																self.adapts, (obs, r, done, info))
			self.adapt_reset = lambda obs, info=None: reduce(lambda obs, adapt: adapt._reset(obs,info), self.adapts, (obs))

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
		self.base_env = {
			"Feeding": ag.FeedingJacoEnv,
			"Laptop": ag.LaptopJacoEnv,
			"OneSwitch": ag.OneSwitchJacoEnv,
			"ThreeSwitch": ag.ThreeSwitchJacoEnv,
			"AnySwitch": ag.AnySwitchJacoEnv,
			"Bottle": ag.BottleJacoEnv,
			"Kitchen": ag.KitchenJacoEnv,
		}[config['env_name']]
		self.base_env = self.base_env(**config['env_kwargs'])
		self.observation_space = self.base_env.observation_space
		self.action_space = self.base_env.action_space
		self.feature_sizes = self.base_env.feature_sizes
		self.terminate_on_failure = config['terminate_on_failure']

	def step(self, action):
		obs, r, done, info = self.base_env.step(action)
		done = info['task_success'] or (self.terminate_on_failure and self.base_env.wrong_goal_reached())
		return obs, r, done, info

	def reset(self):
		return self.base_env.reset()

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
			return obs, r, done, info

		def reset(self):
			self.action = np.zeros(7)
			return super().reset()

	return Action

def session_factory(base):
	class Session(base):
		def __init__(self, config):
			config['env_kwargs']['session_goal'] = True
			super().__init__(config)
			self.goal_reached = False
		def new_goal(self, index=None):
			self.base_env.set_target_index(index)
			self.base_env.reset_noise()
			self.goal_reached = False
		def step(self,action):
			o,r,d,info = super().step(action)
			if info['task_success']:
				self.goal_reached = True
			return o,r,d,info
		def reset(self):
			return super().reset()
	return Session
		

class array_to_dict:
	def __init__(self,master_env,config):
		pass
	def _step(self,obs,r,done,info):
		if not isinstance(obs,dict):
			obs = {'raw_obs': obs}
		return obs,r,done,info
	def _reset(self,obs,info=None):
		if not isinstance(obs,dict):
			obs = {'raw_obs': obs}
		return obs

class oracle:
	def __init__(self,master_env,config):
		self.oracle_type = config['oracle']
		if 'model' in self.oracle_type:
			self.oracle = master_env.oracle = {
				"Feeding": StraightLineOracle,
				"Laptop": LaptopOracle,
				"OneSwitch": OneSwitchOracle,
				"AnySwitch": OneSwitchOracle,
				"ThreeSwitch": ThreeSwitchOracle,
				"Bottle": BottleOracle,
				"Kitchen": StraightLineOracle,
			}[master_env.env_name](master_env.rng,**config['oracle_kwargs'])
			# if 'sim_gaze' in self.oracle_type:
			# 	self.oracle = master_env.oracle = SimGazeModelOracle(base_oracle=self.oracle,
			# 														 **config['gaze_oracle_kwargs'])
			# elif 'gaze' in self.oracle_type:
			# 	self.oracle = master_env.oracle = RealGazeModelOracle(base_oracle=self.oracle)
		else:
			oracle_type = {
				'keyboard': KeyboardOracle,
				'gaze': RealGazeKeyboardOracle,
				# 'sim_gaze': SimGazeKeyboardOracle,
				'dummy_gaze': BottleOracle,
			# }[config['oracle']](master_env)
			}[config['oracle']]
			if config['oracle'] == 'sim_gaze': #TODO: look at how oracke works (why oracle_type)
				self.oracle = master_env.oracle = oracle_type(**config['gaze_oracle_kwargs'])
			elif config['oracle'] == 'dummy_gaze':
				self.oracle = master_env.oracle = oracle_type(master_env.rng,**config['oracle_kwargs'])
			else:
				self.oracle = master_env.oracle = oracle_type()
		self.master_env = master_env
		del master_env.feature_sizes['goal']
		master_env.feature_sizes['recommend'] = self.oracle.size

	def _step(self,obs,r,done,info):
		self._predict(obs,info)
		return obs,r,done,info

	def _reset(self,obs,info=None):
		self.oracle.reset()
		self.master_env.recommend = obs['recommend'] = np.zeros(self.oracle.size)
		return obs

	def _predict(self,obs,info):
		recommend,_info = self.oracle.get_action(obs,info)
		self.master_env.recommend = obs['recommend'] = info['recommend'] = recommend
		info['noop'] = not self.oracle.status.curr_intervention

class static_gaze:
	def __init__(self,master_env,config):
		self.gaze_dim = config['gaze_dim']
		del master_env.feature_sizes['goal']
		master_env.feature_sizes['gaze_features'] = self.gaze_dim
		self.env_name = master_env.env_name
		self.master_env = master_env
		with h5py.File(os.path.join(str(Path(__file__).resolve().parents[2]),'gaze_capture','gaze_data',config['gaze_path']),'r') as gaze_data:
			self.gaze_dataset = {k:v[()] for k,v in gaze_data.items()}
		self.per_step = True

	def sample_gaze(self,index):
		unique_target_index = index
		data = self.gaze_dataset[str(unique_target_index)]
		return self.master_env.rng.choice(data)

	def _step(self,obs,r,done,info):
		if self.per_step:
			self.static_gaze = self.sample_gaze(self.master_env.base_env.target_indices.index(info['unique_index']))
		obs['gaze_features'] = self.static_gaze
		return obs,r,done,info

	def _reset(self,obs,info=None):
		index = self.master_env.base_env.target_indices.index(self.master_env.base_env.unique_index)
		obs['gaze_features'] = self.static_gaze = self.sample_gaze(index)
		return obs

class goal:
	"""
	Chooses what features from info to add to obs
	"""
	def __init__(self,master_env,config):
		self.env_name = master_env.env_name
		self.master_env = master_env
		self.goal_feat_func = dict(
			Bottle=lambda info: [info['target_pos']] if info['target1_reached'] else [info['target1_pos']],
			OneSwitch=lambda info: [info['switch_pos'][info['target_index']],],
			AnySwitch=lambda info: [info['switch_pos'],]
		)[self.env_name]
		self.hindsight_feat = dict(
			Bottle={'tool_pos': 3},
			OneSwitch={'tool_pos':3,},
			AnySwitch={'tool_pos':3}
		)[self.env_name]
		master_env.feature_sizes['goal'] = master_env.goal_size = self.goal_size = sum(self.hindsight_feat.values())

	def _step(self,obs,r,done,info):
		if self.goal is None:
			self.goal = np.concatenate([np.ravel(state_component) for state_component in self.goal_feat_func(info)])
		# goal_feat = np.concatenate((goal_feat,np.zeros(self.high_dim_size-goal_feat.size)))
		hindsight_feat = np.concatenate([np.ravel(info[state_component]) for state_component in self.hindsight_feat.keys()])
		# hindsight_feat = np.concatenate((hindsight_feat,np.zeros(self.high_dim_size-hindsight_feat.size)))
		obs['goal'] = self.goal.copy()
		obs['hindsight_goal'] = hindsight_feat
		return obs,r,done,info

	def _reset(self,obs,info=None):
		self.goal = None
		obs['goal'] = np.zeros(self.goal_size)
		obs['hindsight_goal'] = np.zeros(self.goal_size)
		return obs

class reward:
	""" rewards capped at 'cap' """

	def __init__(self, master_env, config):
		self.range = (config['reward_min'], config['reward_max'])
		self.master_env = master_env
		self.reward_type = config['reward_type']
		if self.master_env.env_name == 'Bottle' and self.reward_type == 'sparse':
			self.reward_type = 'part_sparse'
		self.reward_temp = config['reward_temp']
		self.reward_offset = config['reward_offset']

	def _step(self, obs, r, done, info):
		if self.reward_type == 'custom':
			r = -1 
			r += np.exp(-norm(info['tool_pos'] - info['target1_pos']))/2
			if info['target1_reached']:
				r = -.5
				r += np.exp(-norm(info['tool_pos'] - info['target_pos']))/2
			if info['task_success']:
				r = 0
		elif self.reward_type == 'custom_switch':
			r = 0
			if not info['task_success']:
				dist = np.linalg.norm(info['tool_pos'] - info['switch_pos'][info['target_index']])
				r = np.exp(-self.reward_temp * dist + np.log(1 + self.reward_offset)) - 1

				# because of reward clipping to -1, essentially makes reward always -1 when wrong switch is flipped
				# r -= np.sum(info['target_string'] != info['current_string'])
		elif self.reward_type == 'sparse':
			r = -1 + info['task_success']
		elif self.reward_type == 'part_sparse':
			r = -1 + .5*(info['task_success']+info['target1_reached'])
		elif self.reward_type == 'terminal_interrupt':
			r = info['noop']
			# done = info['noop']
		elif self.reward_type == 'kitchen':
			r = -1 + sum(info['item_placed']+info['item_reached']+[info['microwave_closed']])/len(info['item_placed']+info['item_reached']+[1])
		else:
			raise Exception

		r = np.clip(r, *self.range)
		return obs, r, done, info

	def _reset(self, obs,info=None):
		return obs
