import pybullet as p
import numpy as np
from collections import deque
from numpy.linalg import norm
from .base_oracles import UserModelOracle

class LaptopOracle(UserModelOracle):
	def _query(self,obs,info):
		base_env = self.base_env

		bad_contact =  info['angle_dir'] < 0
		info['bad_contact'] = bad_contact
		info['ineff_contact'] = 0
		self.bad_contacts.append(bad_contact)

		if info['lid_open']:
			threshold = self.threshold
			target_pos = base_env.target_pos
		elif np.sum(self.bad_contacts) > 2:
			threshold = .5
			target_pos = min(base_env.lid_pos,key=lambda x: norm(base_env.tool_pos-x))
			target_pos += np.array([0,.1,0])
		else:
			threshold = self.threshold
			target_pos = min(base_env.lid_pos,key=lambda x: norm(base_env.tool_pos-x))
			if norm(base_env.tool_pos-target_pos) > .1:
				target_pos += np.array([0,.1,0])
			
		old_traj = target_pos - info['old_tool_pos']
		new_traj = base_env.tool_pos - info['old_tool_pos']
		info['cos_error'] = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))
		criterion = info['cos_error'] < threshold
		info['distance_to_target'] = norm(base_env.tool_pos-target_pos)
		return criterion, target_pos
	def reset(self):
		self.bad_contacts = deque(np.zeros(10),10)
		self.ineff_contacts = deque(np.zeros(10),10)
		