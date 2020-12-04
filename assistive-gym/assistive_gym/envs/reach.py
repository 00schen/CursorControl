import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from scipy.signal import sawtooth
from .env import AssistiveEnv

reach_arena = (np.array([-.25,-.5,1]),np.array([.6,.4,.2]))
class ReachEnv(AssistiveEnv):
	def __init__(self, path_type, success_dist=.15, path_length=200, frame_skip=5, robot_type='jaco'):
		super(ReachEnv, self).__init__(robot_type=robot_type, task='switch', frame_skip=frame_skip, time_step=0.02, action_robot_len=7, obs_robot_len=14)
		self.observation_space = spaces.Box(-np.inf,np.inf,(14,), dtype=np.float32)
		self.num_targets = 1
		self.path_type = path_type
		self.path_length=path_length
		self.success_dist = success_dist

	def step(self, action):
		old_target_pos =self.target_pos
		old_tool_pos = self.tool_pos

		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

		new_dist = np.linalg.norm(self.target_pos - self.tool_pos)
		target_traj = old_target_pos - old_tool_pos
		new_traj = self.tool_pos - old_tool_pos
		cos_error = np.dot(target_traj,new_traj)/(norm(target_traj)*norm(new_traj))

		# task_success = new_dist < .01
		# self.task_success += task_success
		self.discrete_frachet += np.linalg.norm(old_target_pos-self.tool_pos) # policy was aiming for old target, but target has also updated
		self.task_success = False
		obs = self._get_obs()

		reward_distance = np.linalg.norm(old_tool_pos-old_target_pos) - new_dist
		reward_action = -np.linalg.norm(action) # Penalize actions
		reward = np.dot([1,.01,1],
			[reward_distance, reward_action, self.task_success])

		info = {
			'task_success': self.task_success,
			'distance_to_target': new_dist,
			'diff_distance': reward_distance,
			'action_size': -reward_action,
			'cos_error': cos_error,
			# 'trajectory': new_traj,
			# 'old_tool_pos': old_tool_pos,
			'frachet': self.discrete_frachet,
			'fraction_t': self.t/self.total_t,
		}
		done = False

		return obs, reward, done, info

	def _get_obs(self):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
		tool_pos = np.array(state[0])
		tool_orient = np.array(state[1]) # Quaternions
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		robot_obs = np.concatenate([tool_pos, tool_orient, robot_joint_positions]).ravel()
		return robot_obs.ravel()

	def reset(self):
		self.task_success = 0
		self.discrete_frachet = 0

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

		wall_pos,wall_orient = np.array([0,-1.1,1]),[0,0,0,1]
		wall_collision = p.createCollisionShape(p.GEOM_BOX,halfExtents=[1,.1,1])
		wall_visual = p.createVisualShape(p.GEOM_BOX,halfExtents=[1,.1,1])
		self.wall = p.createMultiBody(basePosition=wall_pos,baseOrientation=wall_orient,baseCollisionShapeIndex=wall_collision,baseVisualShapeIndex=wall_visual,physicsClientId=self.id)

		"""set up target and initial robot position"""
		offset = self.np_random.random(3)*.1

		amplitude = self.np_random.uniform(.2,.3)
		def sin(t):
			speed = 400*self.frame_skip

			x = .4*sawtooth((2*(t/speed)-.5)*np.pi,.5)
			z = amplitude*np.sin(2*np.pi*x/.8) + 1
			return np.array([x,-.75,z]) + offset

		radius = self.np_random.uniform(.2,.3)
		def circle(t):
			speed = radius/.3*400*self.frame_skip

			x = radius*np.cos(2*np.pi*(t/speed))
			z = radius*np.sin(2*np.pi*(t/speed)) + 1
			return np.array([x,-.75,z]) + offset

		self.param_target = {
			'sin': sin,
			'circle': circle,
		}[self.path_type]

		self.init_robot_arm(self.param_target(0))
		offset = self.tool_pos - self.param_target(0)
		self.generate_target(self.np_random.choice(self.num_targets))

		"""configure pybullet"""
		p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
		# Enable rendering
		p.resetDebugVisualizerCamera(cameraDistance= .8, cameraYaw=120, cameraPitch=-20, cameraTargetPosition=[0, -.5, 1], physicsClientId=self.id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

		return self._get_obs()

	def init_robot_arm(self,og_init_pos):
		init_pos = self.init_pos = og_init_pos
		init_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
		self.util.ik_random_restarts(self.robot, 11, init_pos, init_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.03, step_sim=True)
		self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
		self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

	def generate_target(self,index): 
		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.target_pos = self.tool_pos
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=self.target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		self.t = 0
		self.total_t = 0
		self.update_targets()

	def update_targets(self):
		self.total_t += 1
		if norm(self.tool_pos-self.target_pos) <= self.success_dist:
			self.t += 1
		target_pos = self.param_target(self.t)
		self.target_pos = np.array(target_pos)
		self.targets = [self.target_pos]
		p.resetBasePositionAndOrientation(self.target, self.target_pos, [0, 0, 0, 1], physicsClientId=self.id)

	@property
	def tool_pos(self):
		return np.array(p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])

class SinJacoEnv(ReachEnv):
	def __init__(self,**kwargs):
		super().__init__('sin',robot_type='jaco',**kwargs)
class CircleJacoEnv(ReachEnv):
	def __init__(self,**kwargs):
		super().__init__('circle',robot_type='jaco',**kwargs)
