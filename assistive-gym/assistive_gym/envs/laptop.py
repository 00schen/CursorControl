import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from .env import AssistiveEnv

LOW_LIMIT = .3
HIGH_LIMIT = 2

class LaptopEnv(AssistiveEnv):
	def __init__(self, robot_type='jaco', success_dist=.05, frame_skip=5):
		super(LaptopEnv, self).__init__(robot_type=robot_type, task='laptop', frame_skip=frame_skip, time_step=0.02, action_robot_len=7, obs_robot_len=18)
		self.observation_space = spaces.Box(-np.inf,np.inf,(15,), dtype=np.float32)
		self.num_targets = 12
		self.success_dist = success_dist

	def step(self, action):
		old_tool_pos = self.tool_pos

		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))
		lever_angle = p.getJointStates(self.laptop, jointIndices=[0], physicsClientId=self.id)[0][0]
		lid_open = lever_angle >= (HIGH_LIMIT-.5)
		if lid_open:
			angle_dir,angle_diff = 0,0
			p.resetJointState(self.laptop, jointIndex=0, targetValue=HIGH_LIMIT, physicsClientId=self.id)
		else:
			angle_dir,angle_diff = self.move_lever(self.laptop)
		lever_reward = -abs(lever_angle-HIGH_LIMIT)
		

		task_success = norm(self.tool_pos-self.target_pos) < self.success_dist and lid_open
		self.task_success = task_success
		obs = self._get_obs([0])

		reward = np.dot([1,1], # ignored in user penalty setting
			[lever_reward,task_success])

		info = {
			'task_success': self.task_success,
			'angle_dir': angle_dir,
			'angle_diff': angle_diff,
			'old_tool_pos': old_tool_pos,
			'ineff_contact': 0,
			'target_index': self.target_index,
			'lid_open': lever_angle >= HIGH_LIMIT-.2,

			'tool_pos': self.tool_pos,
			'target_pos': self.target_pos,
			'lid_pos': self.lid_pos,
			'lever_angle': lever_angle,
		}
		done = False

		return obs, reward, done, info

	def move_lever(self,switch):
		old_j_pos = robot_joint_position = p.getJointStates(switch, jointIndices=[0], physicsClientId=self.id)[0][0]
		contacts = p.getContactPoints(bodyA=self.robot, bodyB=switch, linkIndexB=0, physicsClientId=self.id)
		contacts += p.getContactPoints(bodyA=self.tool, bodyB=switch, linkIndexB=0, physicsClientId=self.id)
		if len(contacts) == 0:
			return 0, 0

		normal = contacts[0][7]
		joint_pos,__ = p.multiplyTransforms(*p.getLinkState(switch,0)[:2], p.getJointInfo(switch,0,self.id)[14], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
		radius = np.array(contacts[0][6]) - np.array(joint_pos)
		c_F = -normal[1]
		k = .01
		w = k*np.sign(c_F)*np.sqrt(abs(c_F))*norm(radius)

		for _ in range(self.frame_skip):
			robot_joint_position += w
		
		robot_joint_position = np.clip(robot_joint_position,LOW_LIMIT,HIGH_LIMIT)
		p.resetJointState(self.laptop, jointIndex=0, targetValue=robot_joint_position, physicsClientId=self.id)

		return w,robot_joint_position-old_j_pos

	def get_total_force(self):
		tool_force = 0
		tool_force_at_target = 0
		target_contact_pos = None
		contact_laptop_count = 0
		screen_contact_pos = None
		for c in p.getContactPoints(bodyA=self.tool, physicsClientId=self.id):
			tool_force += c[9]
		for c in p.getContactPoints(bodyA=self.tool, bodyB=self.laptop, physicsClientId=self.id):
			linkA = c[3]
			contact_position = c[6]
			# Enforce that contact is close to the target location
			if linkA in [0,1] and np.linalg.norm(contact_position - self.target_pos) < 0.025:
				tool_force_at_target += c[9]
				target_contact_pos = contact_position
			else:
				contact_laptop_count += 1
		for c in p.getContactPoints(bodyA=self.robot, bodyB=self.laptop, physicsClientId=self.id):
			linkA = c[3]
			contact_position = c[6]
			contact_laptop_count += 1
		return tool_force, tool_force_at_target, target_contact_pos, contact_laptop_count

	def _get_obs(self, forces):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
		tool_pos = np.array(state[0])
		tool_orient = np.array(state[1]) # Quaternions
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		screen_pos = np.array(p.getLinkState(self.laptop, 0, computeForwardKinematics=True,physicsClientId=self.id)[0])
		lever_angle = p.getJointStates(self.laptop, jointIndices=[0], physicsClientId=self.id)[0][0]

		# robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, robot_joint_positions, screen_pos, forces]).ravel()
		# print(robot_joint_positions)
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
		p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
		base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
		self.human_controllable_joint_indices = []
		self.human_lower_limits = np.array([])
		self.human_upper_limits = np.array([])

		"""set up laptop environment objects"""
		table_pos = np.array([.0,-1.1,0])
		self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'), basePosition=table_pos, baseOrientation=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
		laptop_scale = 0.12
		laptop_pos = self.laptop_pos = table_pos+np.array([0,.2,.7])+np.array([.3,.1,0])*self.np_random.uniform(-1,1,3)
		self.laptop = p.loadURDF(os.path.join(self.world_creation.directory, 'laptop', 'laptop.urdf'), basePosition=laptop_pos, baseOrientation=p.getQuaternionFromEuler([0, 0, -np.pi/2], physicsClientId=self.id),\
			 physicsClientId=self.id, globalScaling=laptop_scale, useFixedBase=True)
		p.setCollisionFilterPair(self.laptop, self.laptop, 0, -1, 0, physicsClientId=self.id)
		p.resetJointState(self.laptop, jointIndex=0, targetValue=.35, physicsClientId=self.id)

		"""set up target and initial robot position"""
		self.set_target_index() # instance override in demos
		self.generate_target()
		self.init_robot_arm()

		"""configure pybullet"""
		p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
		# Enable rendering
		p.resetDebugVisualizerCamera(cameraDistance= .6, cameraYaw=150, cameraPitch=-60, cameraTargetPosition=[-.1, 0, .9], physicsClientId=self.id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

		return self._get_obs([0])

	def init_start_pos(self):
		"""exchange this function for curriculum"""
		laptop_pos, _orient = p.getBasePositionAndOrientation(self.laptop, physicsClientId=self.id)
		self.init_pos = laptop_pos + np.array([-.1,0,.4]) + self.np_random.uniform([-0.3,-0.3,0], [0.3,0.3,0], size=3)

	def init_robot_arm(self):
		self.init_start_pos()
		init_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
		best_ik_joints = np.array([-1.0, 3.0, 0.5, 4.0, 0.0, 1.5, 1.0])
		self.util.ik_random_restarts(self.robot, 11, self.init_pos, init_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits,
									best_ik_joints=best_ik_joints, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=100, max_ik_random_restarts=10, random_restart_threshold=0.03, step_sim=True)
		self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
		self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

	def set_target_index(self):
		self.target_index = self.np_random.choice(self.num_targets)

	def generate_target(self): 
		lbody_pos, lbody_orient = p.getBasePositionAndOrientation(self.laptop, physicsClientId=self.id)
		buttons = self.buttons = np.array([0,0,.05]) + np.array(list(product(np.linspace(-.1,.1,3),np.linspace(-.15,.15,4),[0])))
		target_pos, target_orient = p.multiplyTransforms(lbody_pos, lbody_orient, buttons[self.target_index], [0, 0, 0, 1], physicsClientId=self.id)
		
		lbody_pos, lbody_orient = p.getBasePositionAndOrientation(self.laptop, physicsClientId=self.id)
		self.targets = [p.multiplyTransforms(lbody_pos, lbody_orient, target_pos, [0, 0, 0, 1])[0] for target_pos in self.buttons]
		target_pos = self.target_pos = np.array(self.targets[self.target_index])

		sphere_collision = -1
		sphere_visual1 = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=.025, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.valids = [p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual1, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)\
						for target_pos in self.targets]
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=self.success_dist, rgbaColor=[1, 1, 1, 1], physicsClientId=self.id)
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)
		p.resetBasePositionAndOrientation(self.target, target_pos, [0, 0, 0, 1], physicsClientId=self.id)

		self.lids = [p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[-10,-10,-10],
							useMaximalCoordinates=False, physicsClientId=self.id) for _i in range(3)]

		self.update_targets()

	def update_targets(self):
		self.lid_pos = []
		for i,inc in enumerate(np.linspace(-.15,.15,num=3)):
			lid_pos,lid_orient = p.getLinkState(self.laptop, 0, computeForwardKinematics=True, physicsClientId=self.id)[:2]
			lever_pos = np.array([-.2,inc,0])
			target_pos = np.array(p.multiplyTransforms(lid_pos, lid_orient, lever_pos, [0, 0, 0, 1])[0])
			self.lid_pos.append(target_pos)
				
			p.resetBasePositionAndOrientation(self.lids[i], target_pos, [0, 0, 0, 1], physicsClientId=self.id)

	@property
	def tool_pos(self):
		return np.array(p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])

class LaptopJacoEnv(LaptopEnv):
	def __init__(self,**kwargs):
		super().__init__(robot_type='jaco',**kwargs)
