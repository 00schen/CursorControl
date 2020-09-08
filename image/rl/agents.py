import numpy as np
import pybullet as p
from numpy.linalg import norm
from envs import rng

class Agent:
	def reset(self):
		pass

"""Oracle Agents"""
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

class KeyboardAgent(Agent):
	def __init__(self,env):
		self.size = 6
	def get_action(self,obs,info=None):
		action = np.zeros(self.size)
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
				action = inputs[key]
				label = prints[key]
				print(label)
		if not np.count_nonzero(action):
			label = 'noop'
			print('noop')
		return action, {}

class MouseAgent:
	def __init__(self,env):
		self.size = 6
	def get_action(self,obs,info=None):
		action = np.zeros(self.size)
		mouse_event = p.getMouseEvents()[-1]
		new_mouse_pos = np.array([mouse_event[1],mouse_event[2]])
		radians = np.arctan2(*(new_mouse_pos-self.og_mouse_pos))-np.pi/12
		index = np.digitize([radians],np.linspace(0,2*np.pi,6,endpoint=False))
		inputs = {
			3: 	np.array([0,1,0,0,0,0]),
			0: 	np.array([1,0,0,0,0,0]),
			1:	np.array([0,0,1,0,0,0]),
			4:	np.array([0,0,0,1,0,0]),
			2:	np.array([0,0,0,0,0,1]),
			5:	np.array([0,0,0,0,1,0])
		}
		prints = {
			3: 	'left',
			0: 	'right',
			1:	'forward',
			4:	'backward',
			2:	'up',
			5:	'down'
		}
		if norm(new_mouse_pos-self.og_mouse_pos) > .1:
			action = inputs[index]
			label = prints[index]
			print(label)
		if not np.count_nonzero(action):
			label = 'noop'
			print('noop')
		return action, {}

		def reset(self):
			print(p.getMouseEvents())
			mouse_event = p.getMouseEvents()[-1]
			self.og_mouse_pos = np.array([mouse_event[1],mouse_event[2]])

class UserModelAgent:
	def __init__(self,env):
		self.env = env
		self.size = 6
	def get_action(self,obs,info=None):
		if self.prev_noop:
			prob = .4*(1-info['cos_error'])
		else:
			prob = .8 if info['cos_error'] < .25 else .3
		# prob=1
		action = np.zeros(self.size)
		if rng.random() < prob:
			# traj = self.env.target_pos-self.env.tool_pos
			# axis = np.argmax(np.abs(traj))
			# action[2*axis+(traj[axis]>0)] = 1
			action[:3] = self.env.target_pos-self.env.tool_pos
		self.prev_noop = not np.count_nonzero(action)
		return action, {}
	def reset(self):
		self.prev_noop = True

"""Demonstration Agents"""
class FollowerAgent:
	def __init__(self,env):
		self.env = env
	def get_action(self,obs,info=None):
		recommend = obs[-6:]
		if np.count_nonzero(recommend):
			# index = np.argmax(recommend)
			traj = recommend[:3]
			axis = np.argmax(np.abs(traj))
			index = 2*axis+(traj[axis]>0)
			self.trajectory = {
				0: np.array([-1,0,0]),
				1: np.array([1,0,0]),
				2: np.array([0,-1,0]),
				3: np.array([0,1,0]),
				4: np.array([0,0,-1]),
				5: np.array([0,0,1]),
			}[index]
			self.action_index = index
			# self.action_count = 0
		action = np.zeros(6)
		action[self.action_index] = 1
		# self.action_count += 1
		# if self.action_count >= 10:
		# 	self.trajectory = np.array([0,0,0])
		# self.trajectory = self.env.target_pos - self.env.tool_pos
		return action, {"action_index": self.action_index, "trajectory": self.trajectory}
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
		return action, {"action_index": self.action_index, "trajectory": trajectory}
	def reset(self):
		self.action_index = 0

class DemonstrationAgent:
	def __init__(self,env,lower_p=.5):
		self.agents = [FollowerAgent(env.env),EpsilonAgent(env.env,epsilon=1/10)]
		self.lower_p = lower_p
	def get_action(self,obs,info=None):
		p = [self.p,1-self.p]
		actions = [agent.get_action(obs) for agent in self.agents]
		# action,agent_info = rng.choice([agent.get_action(obs) for agent in self.agents],p=p)
		action,agent_info = rng.choice(actions,p=p)
		return action,agent_info
	def reset(self):
		self.p = rng.random()*(1-self.lower_p) + self.lower_p
		# self.p = 1
		for agent in self.agents:
			agent.reset()