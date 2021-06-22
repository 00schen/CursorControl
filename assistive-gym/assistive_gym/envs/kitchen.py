import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from .env import AssistiveEnv
from copy import deepcopy
from collections import OrderedDict

reach_arena = (np.array([-.25,-.5,1]),np.array([.6,.4,.2]))
default_orientation = p.getQuaternionFromEuler([0, 0, 0])
class KitchenEnv(AssistiveEnv):
	def __init__(self, robot_type='jaco',success_dist=.05, session_goal=False, frame_skip=5, capture_frames=False, stochastic=True, debug=False,
					num_targets=4, joint_in_state=False, step_limit=1200, pretrain_assistance=False, target_indices=None):
		super(KitchenEnv, self).__init__(robot_type=robot_type, task='reaching', frame_skip=frame_skip, time_step=0.02, action_robot_len=7, obs_robot_len=14)
		self.observation_space = spaces.Box(-np.inf,np.inf,(7+6+6+2+6,), dtype=np.float32)
		self.num_targets = 2*2*2
		self.num_second_targets = 2
		self.success_dist = success_dist
		self.debug = debug
		self.stochastic = stochastic
		self.goal_feat_sizes = OrderedDict({'bowl_pos': 3, 'door_orders': 2, 'tasks': 6})
		self.session_goal = session_goal
		self.pretrain_assistance = pretrain_assistance

		if target_indices is None:
			self.target_indices = list(np.arange(self.num_targets))
		else:
			for i in target_indices:
				assert 0 <= i < self.num_targets
			self.target_indices = target_indices

		self.goal_set_shape = (2, 3)

	def step(self, action):
		old_tool_pos = self.tool_pos

		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))
		if not self.tasks[3]:
			if self.door_orders[0] == 0 or (self.door_orders[0] == 1 and self.tasks[1]):
				self.pull_microwave()
			if self.door_orders[0] == 1 or (self.door_orders[0] == 0 and self.tasks[0]):
				self.pull_fridge()
		else:
			if self.door_orders[1] == 0 or (self.door_orders[1] == 1 and self.tasks[5]):
				self.pull_microwave()
			if self.door_orders[1] == 1 or (self.door_orders[1] == 0 and self.tasks[4]):
				self.pull_fridge()
				
		microwave_handle = self.microwave_handle
		fridge_handle = self.fridge_handle
		microwave_angle = p.getJointStates(self.microwave, jointIndices=[0], physicsClientId=self.id)[0][0]
		fridge_angle = p.getJointStates(self.fridge, jointIndices=[0], physicsClientId=self.id)[0][0]
		# if food grabbed, cannot pull doors

		if microwave_angle <= -1.3:
			self.tasks[0] = 1
		if fridge_angle >= 1.3:
			self.tasks[1] = 1
		if (norm(self.bowl_pos - self.tool_pos) < .05 and self.tasks[0] and self.tasks[1]):
			self.tasks[2] = 1
		if norm(self.curr_bowl_pos - self.microwave_target_pos) < .1:
			self.tasks[3] = 1
		if self.tasks[3] and microwave_angle > -.05:
			self.tasks[4] = 1
		if self.tasks[3] and fridge_angle < .05:
			self.tasks[5] = 1		
	
		self.task_success = sum(self.tasks) == 6
		if self.assist_number != None:
			self.task_success = sum(self.tasks) >= self.assist_number+1

		reward = 0

		obs = self._get_obs([0])
		info = obs

		done = False

		return obs, reward, done, info

	def pull_microwave(self,):
		handle_pos = self.microwave_handle
		if norm(self.tool_pos-handle_pos) > .04:
			return self.push_microwave()

		old_j_pos = robot_joint_position = p.getJointStates(self.microwave, jointIndices=[0], physicsClientId=self.id)[0][0]
		k = -.01
		for _ in range(self.frame_skip):
			robot_joint_position += k

		robot_joint_position = np.clip(robot_joint_position, -1.7,0)
		p.resetJointState(self.microwave, jointIndex=0, targetValue=robot_joint_position, physicsClientId=self.id)

		return k, robot_joint_position - old_j_pos

	def push_microwave(self):
		old_j_pos = robot_joint_position = p.getJointStates(self.microwave, jointIndices=[0], physicsClientId=self.id)[0][0]
		contacts = p.getContactPoints(bodyA=self.tool, bodyB=self.microwave, linkIndexB=0, physicsClientId=self.id)
		for i in range(7,20):
			contacts += p.getContactPoints(bodyA=self.robot, bodyB=self.microwave, linkIndexA=i, linkIndexB=0, physicsClientId=self.id)
		if len(contacts) == 0:
			return 0, 0

		normal = contacts[0][7]
		joint_pos,__ = p.multiplyTransforms(*p.getLinkState(self.microwave,0)[:2], p.getJointInfo(self.microwave,0,self.id)[14], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
		radius = np.array(contacts[0][6]) - np.array(joint_pos)
		axis,_ = p.multiplyTransforms(np.zeros(3),p.getLinkState(self.microwave,0)[1], [0,0,1], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
		centripedal = np.cross(axis,radius)
		c_F = np.dot(normal,centripedal)/norm(centripedal)
		c_F = -normal[0]
		k = .005
		# k = .007 if robot_joint_position < -.1 else .02
		w = k*np.sign(c_F)*norm(normal)
		if self.tasks[3] == 0:
			w = np.clip(w, 0, 10)
		if self.tasks[3] == 1:
			w = np.clip(w, -10, 0)
		for _ in range(self.frame_skip):
			robot_joint_position -= w
		robot_joint_position = np.clip(robot_joint_position,-1.7,0)
		p.resetJointState(self.microwave, jointIndex=0, targetValue=robot_joint_position, physicsClientId=self.id)
		return k, robot_joint_position - old_j_pos

	def pull_fridge(self,):
		handle_pos = self.fridge_handle		
		if norm(self.tool_pos-handle_pos) > .04 or self.tasks[3] == 1:
			return self.push_fridge()
			# return 0,0

		old_j_pos = robot_joint_position = p.getJointStates(self.fridge, jointIndices=[0], physicsClientId=self.id)[0][0]
		k = .005
		for _ in range(self.frame_skip):
			robot_joint_position += k

		robot_joint_position = np.clip(robot_joint_position, 0, 1.7)
		p.resetJointState(self.fridge, jointIndex=0, targetValue=robot_joint_position, physicsClientId=self.id)

		return k, robot_joint_position - old_j_pos

	def push_fridge(self):
		old_j_pos = robot_joint_position = p.getJointStates(self.fridge, jointIndices=[0], physicsClientId=self.id)[0][0]
		contacts = p.getContactPoints(bodyA=self.tool, bodyB=self.fridge, linkIndexB=0, physicsClientId=self.id)
		for i in range(7,20):
			contacts += p.getContactPoints(bodyA=self.robot, bodyB=self.fridge, linkIndexA=i, linkIndexB=0, physicsClientId=self.id)
		if len(contacts) == 0:
			return 0, 0

		normal = contacts[0][7]
		joint_pos,__ = p.multiplyTransforms(*p.getLinkState(self.fridge,0)[:2], p.getJointInfo(self.fridge,0,self.id)[14], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
		radius = np.array(contacts[0][6]) - np.array(joint_pos)
		axis,_ = p.multiplyTransforms(np.zeros(3),p.getLinkState(self.fridge,0)[1], [0,0,1], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
		centripedal = np.cross(axis,radius)
		c_F = np.dot(normal,centripedal)/norm(centripedal)
		c_F = -normal[0]
		k = .005
		w = k*np.sign(c_F)*norm(normal)
		if self.tasks[3] == 0:
			w = np.clip(w, -10, 0)
		if self.tasks[3] == 1:
			w = np.clip(w, 0, 10)
		for _ in range(self.frame_skip):
			robot_joint_position -= w
		robot_joint_position = np.clip(robot_joint_position, 0, 1.7)
		p.resetJointState(self.fridge, jointIndex=0, targetValue=robot_joint_position, physicsClientId=self.id)
		return k, robot_joint_position - old_j_pos


	def _get_obs(self, forces):
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])

		microwave_angle = p.getJointStates(self.microwave, jointIndices=[0], physicsClientId=self.id)[0][0]
		fridge_angle = p.getJointStates(self.fridge, jointIndices=[0], physicsClientId=self.id)[0][0]

		sub_target = microwave_handle if self.door_orders[0] == 0 else fridge_handle
		if self.tasks[0] or self.tasks[1]:
			sub_target = fridge_handle if self.tasks[0] else microwave_handle
		if self.tasks[0] and self.tasks[1]:
			sub_target = self.bowl_pos
		if self.tasks[2]:
			sub_target = self.microwave_target_pos
		if self.tasks[3]:
			sub_target = microwave_handle if self.door_orders[1] == 0 else fridge_handle
		if self.tasks[4] or self.tasks[5]:
			sub_target = fridge_handle if self.tasks[4] else microwave_handle

		self.goal = np.concatenate((self.bowl_pos.copy(),self.door_orders.copy(),self.tasks.copy()))

		obs = {
			'task_success': self.task_success,
			# 'old_tool_pos': old_tool_pos,
			'target_index': self.target_index,
			'goal': self.goal,

			'tasks': self.tasks.copy(),
			'fridge_pos': self.fridge_pos.copy(),
			'microwave_pos': self.microwave_pos.copy(),
			'fridge_handle': self.fridge_handle,
			'microwave_handle': self.microwave_handle,
			'tool_pos': self.tool_pos,
			'goal_set': [item_pos.copy() for item_pos in self.food_poses],
			'sub_target': sub_target,
			'bowl_pos': self.bowl_pos.copy(),
			'microwave_target_pos': self.microwave_target_pos.copy(),
			'door_orders': self.door_orders.copy(),
			'microwave_angle': microwave_angle,
			'fridge_angle': fridge_angle,

			'joints': robot_joint_positions,
		}
		return obs

	def reset(self):
		"""set up standard environment"""
		self.setup_timing()
		_human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, _human_lower_limits, _human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender\
			 = self.world_creation.create_new_world(furniture_type='wheelchair', init_human=False, static_human_base=True, human_impairment='random', print_joints=False, gender='random')
		self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
		self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
		self.reset_robot_joints()
		wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
		p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
		base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
		self.human_controllable_joint_indices = []
		self.human_lower_limits = np.array([])
		self.human_upper_limits = np.array([])

		"""set up environment objects"""
		table_pos = np.array([.6,-1.25,0])
		table_orient = p.getQuaternionFromEuler([0,0,0])
		self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'), basePosition=table_pos, baseOrientation=table_orient, physicsClientId=self.id)
		self.fridge_pos = table_pos+np.array([-.8,0,.1])
		self.fridge = p.loadURDF(os.path.join(self.world_creation.directory, 'fridge', 'fridge.urdf'), basePosition=self.fridge_pos, baseOrientation=default_orientation,\
			 physicsClientId=self.id, globalScaling=.7, useFixedBase=True)
		table_info = p.getBasePositionAndOrientation(self.table, physicsClientId=self.id)
		microwave_pos, microwave_orient = p.multiplyTransforms(*table_info, np.array([-.45, .3, .75]), p.getQuaternionFromEuler([0,0,np.pi]))
		self.microwave_pos = np.array(microwave_pos)
		print(microwave_pos)
		self.microwave = p.loadURDF(os.path.join(self.world_creation.directory, 'microwave', 'microwave.urdf'), basePosition=microwave_pos, baseOrientation=microwave_orient,\
			 physicsClientId=self.id, globalScaling=.8, useFixedBase=True)
		# for i in range(p.getNumJoints(self.fridge)):
		# 	print(p.getJointInfo(self.fridge,i))

		"""set up target and initial robot position"""
		if not self.session_goal:
			self.set_target_index() # instance override in demos
		self.init_robot_arm()
		self.generate_target()
		self.unique_index = self.target_index//self.num_second_targets
		self._collision_off_hand(self.fridge, 1)
		self._collision_off_hand(self.microwave, 1)

		self.assist_number = None
		if self.pretrain_assistance:
			self.assist_number = assist_number = np.random.randint(6)
			if self.debug:
				self.assist_number = assist_number = 0
			if assist_number == 1 or assist_number > 2:
				if self.door_orders[0] == 0:
					p.resetJointState(self.microwave, jointIndex=0, targetValue=-1.31, physicsClientId=self.id)
					self.tasks[0] = 1
				else:
					p.resetJointState(self.fridge, jointIndex=0, targetValue=1.31, physicsClientId=self.id)
					self.tasks[1] = 1
			if assist_number >= 2:
				if self.door_orders[0] == 0:
					p.resetJointState(self.fridge, jointIndex=0, targetValue=1.31, physicsClientId=self.id)
					self.tasks[1] = 1
				else:
					p.resetJointState(self.microwave, jointIndex=0, targetValue=-1.31, physicsClientId=self.id)
					self.tasks[0] = 1
			if assist_number == 3:
				bowl_orient = p.getBasePositionAndOrientation(self.bowl, physicsClientId=self.id)[1]
				p.resetBasePositionAndOrientation(self.bowl, self.tool_pos-np.array([0,0,.05]), bowl_orient, physicsClientId=self.id)
				self.tasks[2] = 1
			if assist_number >= 4:
				bowl_orient = p.getBasePositionAndOrientation(self.bowl, physicsClientId=self.id)[1]
				p.resetBasePositionAndOrientation(self.bowl, self.microwave_target_pos-np.array([0,0,.05]), bowl_orient, physicsClientId=self.id)
				self.tasks[2] = 1
				self.tasks[3] = 1
			if assist_number == 5:
				if self.door_orders[1] == 0:
					p.resetJointState(self.microwave, jointIndex=0, targetValue=0, physicsClientId=self.id)
					self.tasks[4] = 1
				else:
					p.resetJointState(self.fridge, jointIndex=0, targetValue=0, physicsClientId=self.id)
					self.tasks[5] = 1


		"""configure pybullet"""
		# p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, physicsClientId=self.id)
		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
		# Enable rendering
		p.resetDebugVisualizerCamera(cameraDistance= .7, cameraYaw=180, cameraPitch=-30, cameraTargetPosition=[0, -.4, 1.1], physicsClientId=self.id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

		self.task_success = 0
		return self._get_obs([0])

	def _collision_off_hand(self,obj,ind):
		for i in range(7,20):
			p.setCollisionFilterPair(obj, self.robot, ind, i, 0, physicsClientId=self.id)
		for i in range(-1,2):
			p.setCollisionFilterPair(obj, self.tool, ind, i, 0, physicsClientId=self.id)

	def init_start_pos(self):
		"""exchange this function for curriculum"""
		self.init_pos = np.array([0,-.5,1])
		if self.stochastic:
			self.init_pos += self.np_random.uniform([-0.1,-0.1,0], [0.1,0.1,0], size=3)

	def init_robot_arm(self):
		self.init_start_pos()
		init_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
		best_ik_joints = np.array([-1.0, 3.0, 1, 4.0, 0.0, 1.5, 1.0])
		self.util.ik_random_restarts(self.robot, 11, self.init_pos, init_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits,
									best_ik_joints=best_ik_joints, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1, max_ik_random_restarts=1, random_restart_threshold=0.03, step_sim=True)
		self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
		self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

	def get_random_target(self):
		return np.concatenate((self.food_poses[np.random.choice(2,)],np.random.randint(2,size=(2,))))

	def set_target_index(self):
		self.target_index = self.np_random.choice(self.target_indices)

	def generate_target(self):
		self.door_orders = [(self.target_index%4)//2,self.target_index%2]
		self.tasks = [0,0,0,0,0,0] # [open fridge, open microwave, get bowl, put bowl in microwave, close microwave, close fridge]

		bowl_ind = self.target_index//4
		other_ind = list(range(2))
		del other_ind[bowl_ind]
		other_ind = np.random.permutation(other_ind)
		self.food_poses = []
		for increment in product([-.25],[.2],[.62,.85]):
			food_pos,food_orient = p.multiplyTransforms(self.fridge_pos, default_orientation, increment, default_orientation, physicsClientId=self.id)
			food_pos = np.array(food_pos) + np.random.uniform([-.1,-.02,0],[.1,.02,0])
			self.food_poses.append(food_pos)

		bowl_pos = self.food_poses[bowl_ind]
		self.bowl_pos = bowl_pos + np.array([0,0,.05])
		self.bowl = p.loadURDF(os.path.join(self.world_creation.directory, 'dinnerware', 'bowl.urdf'), basePosition=bowl_pos,baseOrientation=p.getQuaternionFromEuler([np.pi/2,0,0], physicsClientId=self.id),\
			 physicsClientId=self.id, globalScaling=.8, useFixedBase=True)
		self.items = [self.bowl,]
		for i in range(1):
			bottle_pos = self.food_poses[other_ind[i]]
			bottle = p.loadURDF(os.path.join(self.world_creation.directory, 'bottle', 'bottle.urdf'),
								basePosition=bottle_pos, useFixedBase=True, baseOrientation=default_orientation, globalScaling=.01, physicsClientId=self.id)
			self.items.append(bottle)
		# for item in self.items:
		self._collision_off_hand(self.bowl,-1)

		self.microwave_target_pos = self.microwave_pos + np.array([.05,.05,.15])
		handle_pos = self.fridge_handle

		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual,
						 basePosition=self.microwave_target_pos, useMaximalCoordinates=False, physicsClientId=self.id)	
		p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual,
						 basePosition=handle_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		self.update_targets()

	def update_targets(self):
		bowl_orient = p.getBasePositionAndOrientation(self.bowl, physicsClientId=self.id)[1]
		if self.tasks[2] and not self.tasks[3]:
			p.resetBasePositionAndOrientation(self.bowl, self.tool_pos-np.array([0,0,.05]), bowl_orient, physicsClientId=self.id)
			# self.bowl_constraint = p.createConstraint(self.robot, 8, self.bowl, -1, p.JOINT_FIXED, [0, 0, 0], parentFramePosition=[0, 0, 0.02], childFramePosition=[0, 0, .05], 
											# parentFrameOrientation=p.getQuaternionFromEuler([0, -np.pi/2.0, 0]), physicsClientId=self.id)
			# p.changeConstraint(self.bowl_constraint, maxForce=500, physicsClientId=self.id)
		if self.tasks[3]:
			# p.removeConstraint(self.bowl_constraint, physicsClientId=self.id)
			p.resetBasePositionAndOrientation(self.bowl, self.microwave_target_pos-np.array([0,0,.05]), bowl_orient, physicsClientId=self.id)

	@property
	def tool_pos(self):
		return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[0])

	@property
	def tool_orient(self):
		return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[1])

	@property
	def curr_bowl_pos(self):
		return np.array(p.getBasePositionAndOrientation(self.bowl, physicsClientId=self.id)[0])

	@property
	def fridge_handle(self):
		handle_info = p.getLinkState(self.fridge, 1)[:2]
		fridge_handle, _ = p.multiplyTransforms(*handle_info, np.array([.35,.08,.05]),
											 p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.id)
		return fridge_handle
	
	@property
	def microwave_handle(self):
		handle_info = p.getLinkState(self.microwave, 0)[:2]
		microwave_handle, _ = p.multiplyTransforms(*handle_info, np.array([.34,-.05,.12]),
											 p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.id)
		return microwave_handle

class KitchenJacoEnv(KitchenEnv):
	def __init__(self,**kwargs):
		super().__init__(robot_type='jaco',**kwargs)
