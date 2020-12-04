import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from .env import AssistiveEnv

class FeedingEnv(AssistiveEnv):
	def __init__(self, robot_type='pr2', human_control=False, success_dist=.05, frame_skip=10):
		super(FeedingEnv, self).__init__(robot_type=robot_type, task='feeding', human_control=human_control, frame_skip=frame_skip, time_step=0.02, action_robot_len=7, action_human_len=(4 if human_control else 0), obs_robot_len=25, obs_human_len=(23 if human_control else 0))
		self.observation_space = spaces.Box(-np.inf,np.inf,(15,), dtype=np.float32)
		self.num_targets = 1
		self.success_dist = success_dist

	def step(self, action):
		old_dist = np.linalg.norm(self.target_pos - self.tool_pos)
		old_tool_pos = self.tool_pos
		old_traj = self.target_pos - old_tool_pos

		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

		new_dist = np.linalg.norm(self.target_pos - self.tool_pos)
		new_traj = self.tool_pos - old_tool_pos
		cos_error = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))

		robot_force_on_human, spoon_force_on_human = self.get_total_force()
		obs = self._get_obs([spoon_force_on_human], [robot_force_on_human, spoon_force_on_human])
		task_success = new_dist < self.success_dist
		self.task_success += task_success

		# reward_food, food_mouth_velocities, food_hit_human_reward = self.get_food_rewards()
		reward_distance = old_dist - new_dist
		reward_action = -np.linalg.norm(action) # Penalize actions
		reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action
			#  + self.config('food_reward_weight')*reward_food + preferences_score

		info = {
			'task_success': self.task_success,
			'distance_to_target': new_dist,
			'diff_distance': reward_distance,
			# 'action_size': -reward_action,
			'cos_error': cos_error,
			# 'trajectory': new_traj,
			# 'old_tool_pos': old_tool_pos,
		}
		done = False

		return obs, reward, done, info

	def get_total_force(self):
		robot_force_on_human = 0
		spoon_force_on_human = 0
		for c in p.getContactPoints(bodyA=self.robot, bodyB=self.human, physicsClientId=self.id):
			robot_force_on_human += c[9]
		for c in p.getContactPoints(bodyA=self.spoon, bodyB=self.human, physicsClientId=self.id):
			spoon_force_on_human += c[9]
		return robot_force_on_human, spoon_force_on_human

	def get_food_rewards(self):
		# Check all food particles to see if they have left the spoon or entered the person's mouth
		# Give the robot a reward or penalty depending on food particle status
		food_reward = 0
		food_hit_human_reward = 0
		food_mouth_velocities = []
		foods_to_remove = []
		for f in self.foods:
			food_pos, food_orient = p.getBasePositionAndOrientation(f, physicsClientId=self.id)
			distance_to_mouth = np.linalg.norm(self.target_pos - food_pos)
			if distance_to_mouth < 0.02:
				# Delete particle and give robot a reward
				food_reward += 20
				self.task_success += 1
				food_velocity = np.linalg.norm(p.getBaseVelocity(f, physicsClientId=self.id)[0])
				food_mouth_velocities.append(food_velocity)
				foods_to_remove.append(f)
				p.resetBasePositionAndOrientation(f, self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1], physicsClientId=self.id)
				continue
			# elif food_pos[-1] < 0.5 or len(p.getContactPoints(bodyA=f, bodyB=self.table, physicsClientId=self.id)) > 0 or len(p.getContactPoints(bodyA=f, bodyB=self.bowl, physicsClientId=self.id)) > 0:
			elif food_pos[-1] < 0.5:
				# Delete particle and give robot a penalty for spilling food
				food_reward -= 5
				foods_to_remove.append(f)
				continue
			if len(p.getContactPoints(bodyA=f, bodyB=self.human, physicsClientId=self.id)) > 0 and f not in self.foods_hit_person:
				# Record that this food particle just hit the person, so that we can penalize the robot
				self.foods_hit_person.append(f)
				food_hit_human_reward -= 1
		self.foods = [f for f in self.foods if f not in foods_to_remove]
		return food_reward, food_mouth_velocities, food_hit_human_reward

	def _get_obs(self, forces, forces_human):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		spoon_pos, spoon_orient = p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)
		robot_right_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_right_arm_joint_indices, physicsClientId=self.id)
		robot_right_joint_positions = np.array([x[0] for x in robot_right_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
		if self.human_control:
			human_pos = np.array(p.getBasePositionAndOrientation(self.human, physicsClientId=self.id)[0])
			human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
			human_joint_positions = np.array([x[0] for x in human_joint_states])

		head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]

		# robot_obs = np.concatenate([spoon_pos-torso_pos, spoon_orient, robot_right_joint_positions, head_pos-torso_pos, head_orient, forces]).ravel()
		robot_obs = np.concatenate([spoon_pos, spoon_orient, robot_right_joint_positions, forces]).ravel()
		if self.human_control:
			human_obs = np.concatenate([spoon_pos-human_pos, spoon_orient, human_joint_positions, head_pos-human_pos, head_orient, forces_human]).ravel()
		else:
			human_obs = []

		return np.concatenate([robot_obs, human_obs]).ravel()

	def reset(self):
		self.task_success = 0

		"""set up standard environment"""
		self.setup_timing()
		self.human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='wheelchair', static_human_base=True, human_impairment='random', print_joints=False, gender='random')
		self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
		self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
		self.reset_robot_joints()
		wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
		p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
		base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
		joints_positions = [(6, np.deg2rad(-90)), (16, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
		joints_positions += [(21, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (22, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (23, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)))]
		# self.human_controllable_joint_indices = [20, 21, 22, 23]
		# self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
		# self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]
		self.human_controllable_joint_indices = []
		self.human_lower_limits = np.array([])
		self.human_upper_limits = np.array([])
		self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices if (self.human_control or self.world_creation.human_impairment == 'tremor') else [], use_static_joints=True, human_reactive_force=None)
		p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89 if self.gender == 'male' else 0.86], [0, 0, 0, 1], physicsClientId=self.id)
		# human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
		# self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])
		self.target_human_joint_positions = np.array([])

		"""place feeding evironment objects"""
		# Place a bowl of food on a table
		self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'), basePosition=[0.35, -0.9, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
		self.bowl_scale = 0.75
		visual_filename = os.path.join(self.world_creation.directory, 'dinnerware', 'bowl_reduced_compressed.obj')
		collision_filename = os.path.join(self.world_creation.directory, 'dinnerware', 'bowl_vhacd.obj')
		bowl_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=[self.bowl_scale]*3, physicsClientId=self.id)
		bowl_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=[self.bowl_scale]*3, physicsClientId=self.id)
		bowl_pos = np.array([-0.15, -0.55, 0.75]) + np.array([self.np_random.uniform(-0.05, 0.05), self.np_random.uniform(-0.05, 0.05), 0])
		self.bowl = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=bowl_collision, baseVisualShapeIndex=bowl_visual, basePosition=bowl_pos, baseOrientation=p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), baseInertialFramePosition=[0, 0.04*self.bowl_scale, 0], useMaximalCoordinates=False, physicsClientId=self.id)
		
		
		"""set up target and initial robot position (objects set up with target)"""
		self.generate_target(0)
		self.init_robot_arm(np.array(bowl_pos) + np.array([0, -0.1, 0.4]))

		"""configure pybullet"""
		p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
		p.setGravity(0, 0, 0, body=self.human, physicsClientId=self.id)
		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
		# Enable rendering
		p.resetDebugVisualizerCamera(cameraDistance=1.10, cameraYaw=40, cameraPitch=-45, cameraTargetPosition=[-0.2, 0, 0.75], physicsClientId=self.id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
		
		"""generate food"""
		spoon_pos, spoon_orient = p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)
		spoon_pos = np.array(spoon_pos)
		food_radius = 0.005
		food_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=food_radius, physicsClientId=self.id)
		food_visual = -1
		food_mass = 0.001
		food_count = 2*2*2
		batch_positions = []
		for i in range(2):
			for j in range(2):
				for k in range(2):
					batch_positions.append(np.array([i*2*food_radius-0.005, j*2*food_radius, k*2*food_radius+0.02]) + spoon_pos)
		last_food_id = p.createMultiBody(baseMass=food_mass, baseCollisionShapeIndex=food_collision, baseVisualShapeIndex=food_visual, basePosition=[0, 0, 0], useMaximalCoordinates=False, batchPositions=batch_positions, physicsClientId=self.id)
		self.foods = list(range(last_food_id-food_count+1, last_food_id+1))
		self.foods_hit_person = []
		self.total_food_count = len(self.foods)
		# Drop food in the spoon
		for _ in range(100):
			p.stepSimulation(physicsClientId=self.id)

		return self._get_obs([0], [0, 0])

	def init_robot_arm(self,og_init_pos):
		init_pos = self.init_pos = og_init_pos
		init_orient = p.getQuaternionFromEuler(np.array([np.pi/2.0, 0, np.pi/2.0]), physicsClientId=self.id)
		self.util.ik_random_restarts(self.robot, 8, init_pos, init_orient, self.world_creation, self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.01, step_sim=True, check_env_collisions=True)
		self.world_creation.set_gripper_open_position(self.robot, position=1.33, left=False, set_instantly=True)
		self.spoon = self.world_creation.init_tool(self.robot, mesh_scale=[0.08]*3, pos_offset=[0.1, -0.0225, 0.03], orient_offset=p.getQuaternionFromEuler([-0.1, -np.pi/2.0, 0], physicsClientId=self.id), left=False, maximal=False)

	def generate_target(self,index): 
		self.target_index = index
		self.mouth_pos = [0, -0.11, 0.03] if self.gender == 'male' else [0, -0.1, 0.03]
		head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]
		target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
		self.target_pos = np.array(target_pos)
		self.targets = [self.target_pos]

		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=self.success_dist, rgbaColor=[0, 1, 0, 1], physicsClientId=self.id)
		self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=self.target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

		self.update_targets()

	def update_targets(self):
		head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]
		target_pos, target_orient = p.multiplyTransforms(head_pos, head_orient, self.mouth_pos, [0, 0, 0, 1], physicsClientId=self.id)
		self.target_pos = np.array(target_pos)
		self.targets = [self.target_pos]
		p.resetBasePositionAndOrientation(self.target, self.target_pos, [0, 0, 0, 1], physicsClientId=self.id)

	@property
	def tool_pos(self):
		return np.array(p.getBasePositionAndOrientation(self.spoon, physicsClientId=self.id)[0])
