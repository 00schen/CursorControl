import pybullet as p
import numpy as np
from collections import deque
from numpy.linalg import norm
from .base_oracles import UserModelOracle

class BottleOracle(UserModelOracle):
	def _query(self,obs,info):
		bad_contact =  norm(info['old_tool_pos']-info['tool_pos']) < .01
		self.bad_contacts.append(bad_contact)
		info['bad_contact'] = bad_contact
		target2_pos = info['target1_pos'] + np.array([0,.3,0])
		if info['target1_reached'] and self.target2_reached:
			threshold = self.threshold
			target_pos = info['target_pos']
		elif info['target1_reached']:
			threshold = 0
			target_pos = target2_pos
		elif sum(self.bad_contacts) > 0:
			threshold = self.threshold
			target_pos = info['target1_pos']+np.array([0,.25,0])
		elif norm(info['tool_pos']-info['target1_pos']) > .3:
			threshold = self.threshold
			target_pos = info['target1_pos']+np.array([0,.25,0])
		else:
			threshold = self.threshold
			target_pos = info['target1_pos']
		if norm(info['bottle_pos']-target2_pos) < .1:
			self.target2_reached = True
			
		old_traj = target_pos - info['old_tool_pos']
		new_traj = info['tool_pos'] - info['old_tool_pos']
		info['cos_error'] = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))
		criterion = info['cos_error'] < threshold
		info['distance_to_target'] = norm(info['tool_pos']-target_pos)
		return criterion, target_pos
	def reset(self):
		self.bad_contacts = deque(np.zeros(5),5)
		self.target2_reached = False
		