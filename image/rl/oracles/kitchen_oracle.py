import pybullet as p
import numpy as np
from collections import deque
from numpy.linalg import norm
from .base_oracles import UserModelOracle

class KitchenOracle(UserModelOracle):
	def _query(self,obs,info):
		tool_pos = obs['base_obs'][:3]
		tasks = info['tasks']
		orders = info['orders']

		if (orders[0] == 0 and not tasks[0]) or tasks[1]:
			target_pos = info['microwave_handle']
			if norm(tool_pos-target_pos) > .23:
				target_pos += np.array([-.2,.1,0])
			# else:
			# 	target_pos += np.array([0,-.05,0])
		if (orders[0] == 1 and not tasks[1]) or tasks[0]:
			target_pos = info['fridge_handle']
			if norm(tool_pos-target_pos) > .23:
				target_pos += np.array([0,.2,0])
		
		if tasks[1]:
			if self.aux_reached == 0:
				target_pos = info['fridge_handle'] + np.array([.1,.2,-.2])
				if norm(tool_pos-target_pos) < .05:
					self.aux_reached += 1
			if self.aux_reached == 1:
				target_pos = info['fridge_handle'] + np.array([.4,.2,-.2])
				if norm(tool_pos-target_pos) < .05:
					self.aux_reached += 1

		if (tasks[0] and tasks[1]) and self.aux_reached == 2 and not tasks[2]:
			target_pos = info['target1_pos']
			if norm(tool_pos-target_pos) > .23:
				target_pos += np.array([0,.2,0])
		if tasks[2]:
			target_pos = info['target_pos']
			if self.aux_reached < 3:
				target_pos += np.array([-.2,.4,0])	
				if norm(tool_pos-target_pos) < .05:
					self.aux_reached += 1
		if tasks[3] and ((orders[1] == 0 and not tasks[4]) or tasks[5]):
			target_pos = info['microwave_handle']
			if self.aux_reached < 4 + 2*orders[1]:
				target_pos += np.array([-.4,.1,0])
				if norm(tool_pos-target_pos) < .05:
					self.aux_reached += 1
			elif self.aux_reached < 5 + 2*orders[1]:
				target_pos += np.array([.1,.1,0])
				if norm(tool_pos-target_pos) < .05:
					self.aux_reached += 1
			else:
				target_pos += np.array([-.05,-.1,0])
		if tasks[3] and ((orders[1] == 1 and not tasks[5]) or tasks[4]):
			target_pos = info['fridge_handle']
			if self.aux_reached < 4 + 2*(1-orders[1]):
				target_pos += np.array([.2,.2,0])
				if norm(tool_pos-target_pos) < .05:
					self.aux_reached += 1
			elif self.aux_reached < 5 + 2*(1-orders[1]):
				target_pos += np.array([-.1,.2,0])
				if norm(tool_pos-target_pos) < .05:
					self.aux_reached += 1
			else:
				target_pos += np.array([.05,-.1,0])

		gen_target(target_pos)
		
		old_traj = target_pos - info['old_tool_pos']
		new_traj = info['tool_pos'] - info['old_tool_pos']
		cos_error = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))
		criterion = cos_error < self.threshold
		# info['distance_to_target'] = norm(info['tool_pos']-target_pos)
		return criterion, target_pos
	def reset(self):
		self.aux_reached = 0

def gen_target(pos):
	sphere_collision = -1
	sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 1, 1],)
	p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual,
					basePosition=pos, useMaximalCoordinates=False, )	
