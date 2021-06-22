import pybullet as p
import numpy as np
from collections import deque
from numpy.linalg import norm
from .base_oracles import UserModelOracle

class BottleOracle(UserModelOracle):
	def _query(self,obs,info):
		target = obs['goal'][:3]
		# target1 = obs['goal'][3:]
		target1 = obs['goal'][:3]
		tool_pos = obs['base_obs'][:3]
		target1_reached = obs['base_obs'][7]
		checkpoint_poses = np.array([-.2,-1.1,0]) + np.array([0,.1,1.1]) + np.array([0,.3, 0])

		if not target1_reached:
			if norm(tool_pos-target1) > .15:
				target_pos = target1 + np.array([0,.15,0])
			else:
				target_pos = target1
			self.first_target = target_pos
		elif not self.checkpoint:
			target_pos = checkpoint_poses
			if norm(tool_pos-checkpoint_poses) < .05:
				self.checkpoint = True
		else:
			if norm(tool_pos-target) > .15:
				target_pos = target + np.array([0,.15,0])
			else:
				target_pos = target

		old_traj = target_pos - info['old_tool_pos']
		new_traj = info['tool_pos'] - info['old_tool_pos']
		cos_error = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))
		criterion = cos_error < self.threshold
		# info['distance_to_target'] = norm(info['tool_pos']-target_pos)
		return criterion, target_pos
	def reset(self):
		self.checkpoint = False