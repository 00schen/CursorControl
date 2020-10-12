import numpy as np
import pybullet as p
from numpy.linalg import norm
from envs import rng

"""Oracle Agents"""
class Agent:
	def __init__(self):
		self.size = 6
	def reset(self):
		pass

class TrajectoryAgent(Agent):
	def __init__(self,env):
		self.env = env
		self.size = 3
	def get_action(self,obs,info=None):
		action = self.env.target_pos - self.env.tool_pos
		return action, {}

class TargetAgent(Agent):
	def __init__(self,env):
		self.env = env
		self.size = 3
	def get_action(self,obs,info=None):
		action = self.env.target_pos
		return action, {}

import pygame as pg
SCREEN_SIZE = 300
class UserInputAgent(Agent):
	def __init__(self,env):
		super().__init__()
		self.env = env
	def get_input(self):
		pass
	def get_action(self,obs,info=None):
		user_info = self.get_input()
		action = {
			'left': 	np.array([0,1,0,0,0,0]),
			'right': 	np.array([1,0,0,0,0,0]),
			'forward':	np.array([0,0,1,0,0,0]),
			'backward':	np.array([0,0,0,1,0,0]),
			'up':		np.array([0,0,0,0,0,1]),
			'down':		np.array([0,0,0,0,1,0]),
			'noop':		np.array([0,0,0,0,0,0])
		}[self.action]
		print(self.action)
		return action, user_info

class KeyboardAgent(UserInputAgent):
	def get_input(self):
		keys = p.getKeyboardEvents()
		inputs = {
			p.B3G_LEFT_ARROW: 	'left',
			p.B3G_RIGHT_ARROW: 	'right',
			ord('e'):		 	'forward',
			ord('d'):		 	'backward',
			p.B3G_UP_ARROW:		'up',
			p.B3G_DOWN_ARROW:	'down'
		}
		self.action = 'noop'
		for key in inputs:
			if key in keys and keys[key]&p.KEY_WAS_TRIGGERED:
				self.action = inputs[key]
			
		return {"action": self.action}

class MouseAgent(UserInputAgent):
	def get_input(self):
		mouse_pos = pg.mouse.get_pos()
		new_mouse_pos = np.array(mouse_pos)-np.array([SCREEN_SIZE//2,SCREEN_SIZE//2])
		# new_mouse_pos = np.array(mouse_pos)-np.array(self.mouse_pos)
		self.mouse_pos = mouse_pos
		radians = (np.arctan2(*new_mouse_pos) - (np.pi/3) + (2*np.pi)) % (2*np.pi)
		index = np.digitize([radians],np.linspace(0,2*np.pi,7,endpoint=True))[0]
		inputs = {
			1:	'right',
			4:	'left',
			2:	'forward',
			5:	'backward',
			3:	'up',
			6:	'down',
		}
		if norm(new_mouse_pos) > 50:
			self.action = inputs[index]
		else:
			self.action = 'noop'
		return {"mouse_pos": mouse_pos, "action": self.action}

	def reset(self):
		self.mouse_pos = pg.mouse.get_pos()

class UserModelAgent(Agent):
	def __init__(self,env,threshold=.5,epsilon=0):
		super().__init__()
		self.env = env
		self.threshold = threshold
		self.epsilon = epsilon
	def get_action(self,obs,info=None):
		if self.prev_noop:
			prob = (1-info['cos_error'])/4
		else:
			prob = info['cos_error'] < self.threshold
		prob = info['cos_error'] < self.threshold
		# prob = 0
		action = np.zeros(self.size)
		if rng.random() < prob:
			traj = self.env.target_pos-self.env.tool_pos
			axis = np.argmax(np.abs(traj))
			if rng.random() > self.epsilon:
				index = 2*axis+(traj[axis]>0)
			else:
				index = rng.integers(6)
			action[index] = 1

			# action[:3] = self.env.target_pos-self.env.tool_pos
		self.prev_noop = not np.count_nonzero(action)
		return action, {}
	def reset(self):
		self.prev_noop = True

"""Demonstration Agents"""
class FollowerAgent:
	def __init__(self,env,traj_len):
		self.env = env
		self.traj_len = traj_len
	def get_action(self,obs,info=None):
		recommend = obs[-6:]
		if np.count_nonzero(recommend):
			index = np.argmax(recommend)
			# traj = recommend[:3]
			# axis = np.argmax(np.abs(traj))
			# index = 2*axis+(traj[axis]>0)
			self.trajectory = {
				0: np.array([-1,0,0]),
				1: np.array([1,0,0]),
				2: np.array([0,-1,0]),
				3: np.array([0,1,0]),
				4: np.array([0,0,-1]),
				5: np.array([0,0,1]),
			}[index]
			self.action_index = index
		# action = np.zeros(6)
		# action[self.action_index] = 1


		action = self.env.target_pos-self.env.tool_pos
		return self.trajectory*self.traj_len, {"action_index": self.action_index, "trajectory": self.trajectory}
	def reset(self):
		self.trajectory = np.array([0,0,0])
		self.action_count = 0
		self.action_index = 0

class EpsilonAgent:
	def __init__(self,env,epsilon=.25):
		self.epsilon = epsilon
		self.env = env
	def get_action(self,obs,info=None):
		real_traj = self.env.target_pos-self.env.tool_pos
		real_index = np.argmax(np.abs(real_traj))
		real_index = real_index*2 + real_traj[real_index] > 0
		if self.action_index == real_index:
			self.action_index = rng.choice(6)
		else:
			self.action_index = self.action_index if rng.random() > self.epsilon else rng.choice(6)
		action = np.zeros(6)
		action[self.action_index] = 1
		trajectory = [
					np.array((-1,0,0)),
					np.array((1,0,0)),
					np.array((0,-1,0)),
					np.array((0,1,0)),
					np.array((0,0,-1)),
					np.array((0,0,1)),
				][self.action_index]

		action = rng.random(3)
		return trajectory, {"action_index": self.action_index, "trajectory": trajectory}
	def reset(self):
		self.action_index = 0

class DemonstrationAgent:
	def __init__(self,env,lower_p=.5,upper_p=1,traj_len=.5):
		self.agents = [FollowerAgent(env.env,traj_len),EpsilonAgent(env.env,epsilon=1/10)]
		self.lower_p = lower_p
		self.upper_p = upper_p
	def get_action(self,obs,info=None):
		p = [self.p,1-self.p]
		actions = [agent.get_action(obs) for agent in self.agents]
		action,agent_info = rng.choice(actions,p=p)
		return action,agent_info
	def reset(self):
		self.p = rng.random()*(self.upper_p-self.lower_p) + self.lower_p
		# self.p = 1
		for agent in self.agents:
			agent.reset()