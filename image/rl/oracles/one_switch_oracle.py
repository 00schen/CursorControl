import pybullet as p
import numpy as np
from collections import deque
from numpy.linalg import norm
from .base_oracles import UserModelOracle

class OneSwitchOracle(UserModelOracle):
	def _query(self,obs,info):
		target_indices = np.nonzero(np.not_equal(info['target_string'],info['current_string']))[0]
		target_poses1 = np.array(info['aux_switch_pos'])[target_indices]
		target_poses = np.array(info['switch_pos'])[target_indices]

		bad_contact =  np.any(np.logical_and(info['angle_dir'] != 0,
							np.logical_or(np.logical_and(info['angle_dir'] < 0, info['target_string'] == 1),
								np.logical_and(info['angle_dir'] > 0, info['target_string'] == 0))))
		info['bad_contact'] = bad_contact
		ineff_contact = (info['ineff_contact'] and np.all(np.abs(info['angle_dir']) < .005))\
						and min(norm(np.array(info['switch_pos'])-info['tool_pos'],axis=1)) < .2
		info['ineff_contact'] = ineff_contact
		self.ineff_contacts.append(ineff_contact)
		self.bad_contacts.append(bad_contact)

		if len(target_indices) == 0:
			info['cos_error'] = 1
			info['distance_to_target'] = 0
			return False, np.zeros(3)
		
		if np.sum(self.bad_contacts) > 5:
			tool_pos = info['tool_pos']
			on_off = info['target_string'][target_indices[0]]

			target_pos2 = [0,.2,.1] if on_off == 0 else [0,.2,-.1]
			target_pos2 = np.array(p.multiplyTransforms(target_poses[0], info['switch_orient'], target_pos2, [0, 0, 0, 1])[0])

			# sphere_collision = -1
			# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=.03, rgbaColor=[0, 1, 1, 1], physicsClientId=base_env.id)
			# target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos2,
			# 					useMaximalCoordinates=False, physicsClientId=base_env.id)

			if ((info['tool_pos'][2] < target_poses[0][2] and on_off == 0)
				or (info['tool_pos'][2] > target_poses[0][2] and on_off == 1)):
				threshold = self.threshold
				target_pos = target_pos2
			elif ((info['tool_pos'][2] > target_poses[0][2] + .15 and on_off == 0)
				or (info['tool_pos'][2] < target_poses[0][2] - .15 and on_off == 1)):
				threshold = 0
				target_pos = target_poses1[0]
			else:
				threshold = .5
				target_pos = tool_pos + np.array([0,0,1]) if on_off == 0 else tool_pos + np.array([0,0,-1])
		elif np.sum(self.ineff_contacts) > 5:
			on_off = info['target_string'][target_indices[0]]
			target_pos = np.array(p.multiplyTransforms(target_poses[0], info['switch_orient'], [0,1,0], [0, 0, 0, 1])[0])
			threshold = .5
		elif norm(info['tool_pos']-target_poses1,axis=1)[0] > .12:
			threshold = self.threshold
			target_pos = target_poses1[0]
		else:
			threshold = .5
			target_pos = target_poses[0]
		# if norm(info['tool_pos']-target_poses1,axis=1)[0] > .12:
		# 	threshold = self.threshold
		# 	target_pos = target_poses1[0]
		# else:
		# 	threshold = self.threshold
		# 	target_poss = target_poses1[0]

			
		old_traj = target_pos - info['old_tool_pos']
		new_traj = info['tool_pos'] - info['old_tool_pos']
		info['cos_error'] = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))
		criterion = info['cos_error'] < threshold
		info['distance_to_target'] = norm(info['tool_pos']-target_pos)
		return criterion, target_pos
	def reset(self):
		self.bad_contacts = deque(np.zeros(10),10)
		self.ineff_contacts = deque(np.zeros(10),10)
		