import pybullet as p
import numpy as np
from collections import deque
from numpy.linalg import norm
from .base_oracles import UserModelOracle

class BottleOracle(UserModelOracle):
	def _query(self,obs,info):
		bad_contact =  np.count_nonzero(info['normal']) > 0
		self.bad_contacts.append(bad_contact)
		info['bad_contact'] = bad_contact
		# info['aux_target_pos'] = target2_pos = info['target1_pos'] + np.array([0,.3,0])
		info['aux_target_pos'] = target2_pos = info['shelf_pos'] + np.array([-.3,.3,0])

		if info['target1_reached'] and self.target2_reached:
			threshold = self.threshold
			target_pos = info['target_pos']
		elif info['target1_reached']:
			threshold = 0
			target_pos = target2_pos

		elif np.count_nonzero(info['normal']) > 0 and not info['target1_reached']:
			threshold = self.threshold
			# if abs(info['normal'][2]) > abs(info['normal'][0]):
				# 
			target_pos = info['normal']
			# sphere_collision = -1
			# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 1, 1],)
			# p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False,)
			# else:
				# target_pos = info['target1_pos'] + np.array([0,.2,0])	
		elif sum(self.bad_contacts) > 0:
			threshold = self.threshold
			# target_pos = info['target1_pos']+np.array([0,.25,0])
			target_pos = np.array([info['tool_pos'][0],info['tool_pos'][1],info['target1_pos'][2]])

		elif norm(info['tool_pos']-info['target1_pos']) > .3:
			target_pos = info['target1_pos']+np.array([0,.25,0])
			threshold = self.threshold
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
		info['bottle_distance'] = norm(info['bottle_pos']-info['target_pos'])
		return criterion, target_pos
	def reset(self):
		self.bad_contacts = deque(np.zeros(5),5)
		self.target2_reached = False
		