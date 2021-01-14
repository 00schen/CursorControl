import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from .env import AssistiveEnv

reach_arena = (np.array([-.25,-.5,1]),np.array([.6,.4,.2]))
class KitchenEnv(AssistiveEnv):
	def __init__(self, robot_type='jaco',success_dist=.05, frame_skip=5):
		super(KitchenEnv, self).__init__(robot_type=robot_type, task='switch', frame_skip=frame_skip, time_step=0.02, action_robot_len=7, obs_robot_len=14)
		self.observation_space = spaces.Box(-np.inf,np.inf,(14,), dtype=np.float32)
		self.num_targets = 6
		self.success_dist = success_dist

	def step(self, action):
		old_tool_pos = self.tool_pos

		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

		task_success = norm(self.tool_pos-self.target_pos) < self.success_dist
		self.task_success = task_success
		obs = self._get_obs([0])

		dist_reward = norm(self.tool_pos-self.target_pos)
		reward = np.dot([1,1], # ignored in user penalty setting
			[dist_reward,task_success])

		info = {
			'task_success': self.task_success,
			'old_tool_pos': old_tool_pos,
			'target_index': self.target_index,

			'tool_pos': self.tool_pos,
			'target_pos': self.target_pos,
		}
		done = False

		return obs, reward, done, info

	def _get_obs(self, forces):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		# state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
		state = p.getLinkState(self.robot, 14, computeForwardKinematics=True, physicsClientId=self.id)
		tool_pos = np.array(state[0])
		tool_orient = np.array(state[1]) # Quaternions
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		robot_obs = np.concatenate([tool_pos, tool_orient, robot_joint_positions, forces]).ravel()
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
		p.resetBasePositionAndOrientation(self.wheelchair, np.array([0,2,0]), wheelchair_orient, physicsClientId=self.id)
		wheelchair_pos = np.array([0,2,0])
		p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
		base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
		self.human_controllable_joint_indices = []
		self.human_lower_limits = np.array([])
		self.human_upper_limits = np.array([])

		"""set up shelf environment objects"""
		default_orientation = p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id)
		self.table = p.loadMJCF(os.path.join(self.world_creation.directory, 'kitchen', 'kitchen.xml'), physicsClientId=self.id)

		"""set up target and initial robot position"""
		self.set_target_index() # instance override in demos
		self.init_robot_arm()
		self.generate_target()

		"""configure pybullet"""
		# p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, physicsClientId=self.id)
		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
		# Enable rendering
		p.resetDebugVisualizerCamera(cameraDistance= .6, cameraYaw=150, cameraPitch=-60, cameraTargetPosition=[-.1, 0, .9], physicsClientId=self.id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

		return self._get_obs([0])

	def init_start_pos(self):
		"""exchange this function for curriculum"""
		self.init_pos = np.array([0,-.5,1])+self.np_random.uniform([-0.1,-0.1,0], [0.1,0.1,0], size=3)

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
		target_pos = self.target_pos = np.zeros(3)
		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		# self.tool = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=self.tool_pos, useMaximalCoordinates=False, physicsClientId=self.id)
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		self.update_targets()

	def update_targets(self):
		pass
		
	@property
	def tool_pos(self):
		return np.array(p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])

class KitchenJacoEnv(KitchenEnv):
	def __init__(self,**kwargs):
		super().__init__(robot_type='jaco',**kwargs)
