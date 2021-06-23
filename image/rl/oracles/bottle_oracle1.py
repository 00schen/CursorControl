import pybullet as p
import numpy as np
from collections import deque
from numpy.linalg import norm
from .base_oracles import Oracle

class BottleOracle(Oracle):
	def _query(self,obs):
		target = obs['target_pos']
		tool_pos = obs['tool_pos']
		door_pos = obs['door_pos']
		shelf_pos = obs['shelf_pos']
		final_door_pos = (np.array([-.15,.17,0]) if obs['target_index']//2 else np.array([.15,.17,0])) + shelf_pos
		door_offset = np.array([.02,0,0]) if obs['target_index']%2 else np.array([-.02,0,0])
		aux_pos = door_pos + np.array([.1,.2,0])
		# door_open = norm(door_pos-final_door_pos) < .05

		if not obs['door_open']:
			if not self.aux_reached and obs['target_index'] == 3:
				target_pos = aux_pos
			else:
				target_pos = door_pos + door_offset
			if norm(tool_pos-aux_pos) < .05:
				self.aux_reached = True
		else:
			if norm(tool_pos-target) > .2:
				target_pos = target + np.array([0,.2,0])
			else:
				target_pos = target

		old_traj = target_pos - obs['old_tool_pos']
		new_traj = obs['tool_pos'] - obs['old_tool_pos']
		cos_error = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))
		criterion = cos_error < self.threshold
		# obs['distance_to_target'] = norm(obs['tool_pos']-target_pos)
		return criterion, target_pos
	def reset(self):
		self.aux_reached = False