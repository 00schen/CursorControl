import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from .env import AssistiveEnv

reach_arena = (np.array([-.25,-.5,1]),np.array([.6,.4,.2]))
class BottleEnv(AssistiveEnv):
	def __init__(self, robot_type='jaco',success_dist=.05, frame_skip=5, capture_frames=False, stochastic=True, debug=False):
		super(BottleEnv, self).__init__(robot_type=robot_type, task='reaching', frame_skip=frame_skip, time_step=0.02, action_robot_len=7, obs_robot_len=14)
		self.observation_space = spaces.Box(-np.inf,np.inf,(7,), dtype=np.float32)
		self.num_targets = 4*2
		self.success_dist = success_dist
		self.debug = debug
		self.stochastic = stochastic

	def step(self, action):
		old_tool_pos = self.tool_pos

		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))
		if norm(self.tool_pos-self.target1_pos) < .05:
			self.target1_reached = True

		contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.shelf, physicsClientId=self.id)
		if len(contacts) == 0:
			normal = np.zeros(3)
		else:
			normal = contacts[0][7]

		task_success = norm(self.target1_pos-self.target_pos) < self.success_dist*2
		self.task_success = task_success
		obs = self._get_obs([0])
		if self.debug:
			self.task_success = task_success = self.target1_reached

		dist_reward = norm(self.tool_pos-self.target1_pos)
		dist1_reward = norm(self.target1_pos-self.target_pos)
		reward = np.dot([1,1,1], # ignored in user penalty setting
			[dist_reward,dist1_reward,task_success])

		info = {
			'task_success': self.task_success,
			'old_tool_pos': old_tool_pos,
			'target_index': self.target_index,
			'normal': normal,

			'shelf_pos': self.shelf_pos,
			'tool_pos': self.tool_pos,
			'bottle_pos': self.target1_pos.copy(),
			'target1_pos': self.og_target1_pos.copy(),
			'target_pos': self.target_pos.copy(),
			'target1_reached': self.target1_reached,
			'unique_index': 4+(self.target_index%2) if self.target1_reached else self.target_index//2
		}
		done = False

		return obs, reward, done, info

	def _get_obs(self, forces):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		state = p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)
		tool_pos = np.array(state[0])
		tool_orient = np.array(state[1]) # Quaternions
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		robot_obs = np.concatenate([tool_pos, tool_orient]).ravel()
		# robot_obs = np.concatenate([tool_pos, tool_orient, robot_joint_positions, forces]).ravel()
		return robot_obs.ravel()

	def reset(self):
		self.task_success = 0

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

		"""set up shelf environment objects"""
		default_orientation = p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id)
		table_pos = np.array([-.3,-1.1,0])
		self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'), basePosition=table_pos, baseOrientation=default_orientation, physicsClientId=self.id)
		shelf_pos = self.shelf_pos = table_pos+np.array([.3,0,.93])
		self.shelf = p.loadURDF(os.path.join(self.world_creation.directory, 'shelf', 'shelf.urdf'), basePosition=shelf_pos, baseOrientation=default_orientation,\
			 physicsClientId=self.id, useFixedBase=True)
		# for i in range(10,20):
		# 	p.setCollisionFilterPair(self.shelf, self.robot, -1, i, 0, physicsClientId=self.id)
		register_pos = self.register_pos = table_pos+np.array([-.4,-.1,.73])
		self.register = p.loadURDF(os.path.join(self.world_creation.directory, 'register', 'register.urdf'), basePosition=register_pos, baseOrientation=default_orientation,\
			 physicsClientId=self.id, useFixedBase=True)
		self.bottles = []
		self.bottle_pos = []
		for increment in product(np.linspace(-.1,.1,num=2),[0],[-.18,.01]):
			bottle_pos,bottle_orient = p.multiplyTransforms(shelf_pos, default_orientation, increment, default_orientation, physicsClientId=self.id)
			bottle = p.loadURDF(os.path.join(self.world_creation.directory, 'bottle', 'bottle.urdf'),
								basePosition=bottle_pos, useFixedBase=True, baseOrientation=bottle_orient, globalScaling=.01, physicsClientId=self.id)
			self.bottles.append(bottle)
			self.bottle_pos.append(bottle_pos+np.array([0,0,.05]))
			# p.setCollisionFilterPair(bottle, self.shelf, -1, -1, 0, physicsClientId=self.id)

		"""set up target and initial robot position"""
		self.set_target_index() # instance override in demos
		self.init_robot_arm()
		self.generate_target()

		"""configure pybullet"""
		# p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, physicsClientId=self.id)
		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
		# Enable rendering
		# p.resetDebugVisualizerCamera(cameraDistance = .6, cameraYaw=150, cameraPitch=-60, cameraTargetPosition=[-.1, 0, .9], physicsClientId=self.id)
		p.resetDebugVisualizerCamera(cameraDistance= .1, cameraYaw=180, cameraPitch=-30, cameraTargetPosition=[0, -.4, 1.1], physicsClientId=self.id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

		return self._get_obs([0])

	def init_start_pos(self):
		"""exchange this function for curriculum"""
		self.init_pos = np.array([0,-.5,1])
		if self.stochastic:
			self.init_pos += self.np_random.uniform([-0.1,-0.1,0], [0.1,0.1,0], size=3)

	def init_robot_arm(self):
		self.init_start_pos()
		init_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
		# best_ik_joints = np.array([-1.0, 3.0, 0.5, 4.0, 0.0, 1.5, 1.0])
		best_ik_joints = np.array([-1.0, 3.0, 1, 4.0, 0.0, 1.5, 1.0])
		self.util.ik_random_restarts(self.robot, 11, self.init_pos, init_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits,
									best_ik_joints=best_ik_joints, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1, max_ik_random_restarts=1, random_restart_threshold=0.03, step_sim=True)
		self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
		self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

	def set_target_index(self):
		self.target_index = self.np_random.choice(self.num_targets)

	def generate_target(self): 
		for bottle in self.bottles:
			for i in range(7,20):
				p.setCollisionFilterPair(bottle, self.robot, -1, i, 0, physicsClientId=self.id)
			for i in range(-1,2):
				p.setCollisionFilterPair(bottle, self.tool, -1, i, 0, physicsClientId=self.id)

		target1_pos = self.target1_pos = self.bottle_pos[self.target_index//2]
		self.og_target1_pos = self.target1_pos.copy()
		self.target1_reached = False
		target_pos = self.target_pos = [self.register_pos + np.array([.2,.2,.05]), self.shelf_pos + np.array([.3,.3,-.1])][self.target_index%2] # get top of shelf
		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.target1 = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target1_pos, useMaximalCoordinates=False, physicsClientId=self.id)
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		self.update_targets()

	def update_targets(self):
		# p.resetBasePositionAndOrientation(self.tool, self.tool_pos, [0, 0, 0, 1], physicsClientId=self.id)
		if self.target1_reached:
			self.target1_pos = self.bottle_pos[self.target_index//2] = self.tool_pos
			p.resetBasePositionAndOrientation(self.target1, self.target1_pos, [0, 0, 0, 1], physicsClientId=self.id)
			p.resetBasePositionAndOrientation(self.bottles[self.target_index//2], self.target1_pos-np.array([0,0,.05]), [0,0,0,1], physicsClientId=self.id)

	@property
	def tool_pos(self):
		# return np.array(p.getLinkState(self.tool, -1, computeForwardKinematics=True, physicsClientId=self.id)[0])
		return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[0])
		# return np.array(p.getLinkState(self.robot, 8, computeForwardKinematics=True, physicsClientId=self.id)[0])

class BottleJacoEnv(BottleEnv):
	def __init__(self,**kwargs):
		super().__init__(robot_type='jaco',**kwargs)
