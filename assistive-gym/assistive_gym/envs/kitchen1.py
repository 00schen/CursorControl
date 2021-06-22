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
	def __init__(self, robot_type='jaco',success_dist=.05, session_goal=False, frame_skip=5, capture_frames=False, stochastic=True, debug=False):
		super(KitchenEnv, self).__init__(robot_type=robot_type, task='reaching', frame_skip=frame_skip, time_step=0.02, action_robot_len=7, obs_robot_len=14)
		self.observation_space = spaces.Box(-np.inf,np.inf,(7,), dtype=np.float32)
		self.num_targets = 3*2
		self.success_dist = success_dist
		self.debug = debug
		self.feature_sizes = OrderedDict({'goal': 7})
		self.stochastic = stochastic
		self.session_goal = session_goal

	def step(self, action):
		microwave_angle = 0
		microwave_closed = False
		old_tool_pos = self.tool_pos
		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))
		self.move_lever(self.microwave, 1, 0)
		for i in range(len(self.items)):
			if norm(self.item_poses[i]-self.place_poses[i]) < .1:
				if i != 0 or microwave_angle < -.1:
					self.item_placed[i] = True
			if norm(self.item_poses[i]-self.tool_pos) < .1 \
				and not self.item_placed[i] \
				and True not in np.logical_xor(self.item_reached,self.item_placed):
				self.item_reached[i] = True

		task_success = np.all(self.item_placed) and microwave_closed
		self.task_success = task_success
		obs = self._get_obs([0])
		# if self.debug:
		# 	self.task_success = task_success = self.item_reached

		reward = sum(self.item_placed+self.item_reached+[microwave_closed])/len(self.item_placed+self.item_reached+[1])

		# self.move_microwave()
		# self.move_slide_cabinet()

		info = {
			'task_success': self.task_success,
			'old_tool_pos': old_tool_pos,
			'target_index': self.target_index,

			'shelf_pos': self.shelf_pos,
			'micro_pos': self.micro_pos,
			'tool_pos': self.tool_pos,
			'item_poses': [item_pos.copy() for item_pos in self.item_poses],
			'og_item_poses': [item_pos.copy() for item_pos in self.og_item_poses],
			'target_poses': [item_pos.copy() for item_pos in self.place_poses],
			'target_pos' : [item_pos.copy() for item_pos in self.place_poses][0],
			'item_reached': self.item_reached.copy(),
			'item_placed': self.item_placed.copy(),
			'microwave_closed': microwave_closed,
			'microwave_angle': microwave_angle,
		}
		done = False

		return obs, reward, done, info

	def move_lever(self, obj, handle_ind, joint_ind):
		handle_info = p.getLinkState(obj, handle_ind)[:2]
		handle_pos, _ = p.multiplyTransforms(*handle_info, np.array([.26,-.05,.12]),
                                             p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.id)
		if norm(self.tool_pos-handle_pos) > .04:
			return 0,0

		old_j_pos = robot_joint_position = p.getJointStates(obj, jointIndices=[joint_ind], physicsClientId=self.id)[0][0]
		k = -.01
		for _ in range(self.frame_skip):
			robot_joint_position += k

		robot_joint_position = np.clip(robot_joint_position, -1.7,0)
		p.resetJointState(obj, jointIndex=joint_ind, targetValue=robot_joint_position, physicsClientId=self.id)

		return k, robot_joint_position - old_j_pos

	def _get_obs(self, forces):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		state = p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		# robot_obs = np.concatenate([tool_pos, tool_orient, robot_joint_positions, forces]).ravel()
		robot_obs = dict(
			base_obs = np.concatenate([self.tool_pos, self.tool_orient,]),
			hindsight_goal = np.concatenate([self.tool_pos,]),
			goal = self.goal,
		)
		return robot_obs

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
		table_pos = np.array([-.7,-1.1,0])
		self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'), basePosition=table_pos, baseOrientation=default_orientation, physicsClientId=self.id)
		shelf_pos = self.shelf_pos = table_pos+np.array([.3,.1,.9])
		self.shelf = p.loadURDF(os.path.join(self.world_creation.directory, 'shelf', 'shelf_front.urdf'), basePosition=self.shelf_pos, baseOrientation=default_orientation,\
			 physicsClientId=self.id, globalScaling=1, useFixedBase=True)
		self.oven_pos = table_pos+np.array([1.15,0,.4])
		self.oven = p.loadURDF(os.path.join(self.world_creation.directory, 'oven', 'oven.urdf'), basePosition=self.oven_pos, baseOrientation=p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2], physicsClientId=self.id),\
			 physicsClientId=self.id, globalScaling=.006, useFixedBase=True)
		
		micro_pos = self.micro_pos = shelf_pos+np.array([0, 0, .15])
		self.microwave = p.loadURDF(os.path.join(self.world_creation.directory, 'microwave', 'microwave.urdf'), basePosition=micro_pos, baseOrientation=p.getQuaternionFromEuler([0,0,np.pi], physicsClientId=self.id),\
			 physicsClientId=self.id, globalScaling=.6, useFixedBase=True)
		for i in range(3):
			print(p.getJointInfo(self.microwave,i))
		
		panfood_pos = shelf_pos+np.array([.12,0,-.1])
		self.panfood = p.loadURDF(os.path.join(self.world_creation.directory, 'panfood', 'panfood.urdf'),
								basePosition=panfood_pos, useFixedBase=True,  physicsClientId=self.id)
		bowl_pos = shelf_pos+np.array([-.12,0,-.1])
		self.bowl = p.loadURDF(os.path.join(self.world_creation.directory, 'dinnerware', 'bowl.urdf'), basePosition=bowl_pos,baseOrientation=p.getQuaternionFromEuler([np.pi/2,0,0], physicsClientId=self.id),\
			 physicsClientId=self.id, globalScaling=.8, useFixedBase=True)
		self.item_poses = [bowl_pos+np.array([0,0,.02]),panfood_pos]
		self.place_poses = [self.micro_pos, self.oven_pos+np.array([.1,.1,.55])]
		self.items = [self.bowl, self.panfood]

		"""set up target and initial robot position"""
		if not self.session_goal:
			self.set_target_index() # instance override in demos
		self.init_robot_arm()
		self.generate_target()
		self._collision_off_hand(self.microwave, 1)
		self._collision_off_hand(self.microwave, -1)

		"""configure pybullet"""
		# p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, physicsClientId=self.id)
		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
		# Enable rendering
		p.resetDebugVisualizerCamera(cameraDistance= .8, cameraYaw=150, cameraPitch=-60, cameraTargetPosition=[-.1, 0, .9], physicsClientId=self.id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

		self.task_success = 0
		self.goal = np.array([1,1,1])
		return self._get_obs([0])

	def _collision_off_hand(self,obj,ind):
		for i in range(7,20):
			p.setCollisionFilterPair(obj, self.robot, ind, i, 0, physicsClientId=self.id)
		for i in range(-1,2):
			p.setCollisionFilterPair(obj, self.tool, ind, i, 0, physicsClientId=self.id)

	def init_start_pos(self):
		"""exchange this function for curriculum"""
		self.init_pos = np.array([0,-.5,1])+self.np_random.uniform([-0.1,-0.1,0], [0.1,0.1,0], size=3)

	def init_robot_arm(self):
		self.init_start_pos()
		init_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
		best_ik_joints = np.array([-1.0, 3.0, 1, 4.0, 0.0, 1.5, 1.0])
		self.util.ik_random_restarts(self.robot, 11, self.init_pos, init_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits,
									best_ik_joints=best_ik_joints, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1, max_ik_random_restarts=1, random_restart_threshold=0.03, step_sim=True)
		self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
		self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

	def set_target_index(self):
		self.target_index = self.np_random.choice(self.num_targets)

	def generate_target(self): 
		for food in [self.panfood,self.bowl]:
			for i in range(7,20):
				p.setCollisionFilterPair(food, self.robot, -1, i, 0, physicsClientId=self.id)
			for i in range(-1,2):
				p.setCollisionFilterPair(food, self.tool, -1, i, 0, physicsClientId=self.id)

		# item_pos = self.item_pos = self.item_poses[self.target_index]
		self.og_item_poses = [item_pos.copy() for item_pos in self.item_poses]
		self.item_reached = [False,False]
		self.item_placed = [False,False] # bowl in microwave, panfood in pan
		# target_poses = self.target_poses = self.place_poses[self.target_index]
		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.item = [p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual,
						 basePosition=item_pos, useMaximalCoordinates=False, physicsClientId=self.id) for item_pos in self.item_poses]
		self.target = [p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual,
						 basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id) for target_pos in self.place_poses]	
		self.update_targets()
		# joint_pos,__ = p.multiplyTransforms(*p.getBasePositionAndOrientation(self.microwave,0)[:2], p.getJointInfo(self.microwave,0,self.id)[14], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
		# p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=joint_pos, useMaximalCoordinates=False, physicsClientId=self.id)

	def update_targets(self):
		for i,holding in enumerate(np.logical_xor(self.item_reached,self.item_placed)):
			if holding:
				self.item_poses[i] = self.tool_pos
				item_orienation = p.getBasePositionAndOrientation(self.items[i], physicsClientId=self.id)[1]
				p.resetBasePositionAndOrientation(self.item[i], self.tool_pos, [0, 0, 0, 1], physicsClientId=self.id)
				p.resetBasePositionAndOrientation(self.items[i], self.tool_pos, item_orienation, physicsClientId=self.id)
		for i, placed in enumerate(self.item_placed):
			if placed:
				self.item_poses[i] = self.place_poses[i]
				item_orienation = p.getBasePositionAndOrientation(self.items[i], physicsClientId=self.id)[1]
				p.resetBasePositionAndOrientation(self.item[i], self.place_poses[i], [0, 0, 0, 1], physicsClientId=self.id)
				p.resetBasePositionAndOrientation(self.items[i], self.place_poses[i], item_orienation, physicsClientId=self.id)
	@property
	def tool_pos(self):
		return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[0])
	
	@property
	def tool_orient(self):
		return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[1])

class KitchenJacoEnv(KitchenEnv):
	def __init__(self,**kwargs):
		super().__init__(robot_type='jaco',**kwargs)
