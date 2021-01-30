import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from .env import AssistiveEnv

LOW_LIMIT = -1
HIGH_LIMIT = .2

class LightSwitchEnv(AssistiveEnv):
	def __init__(self,message_indices,success_dist=.05,frame_skip=5,robot_type='jaco',capture_frames=False):
		super(LightSwitchEnv, self).__init__(robot_type=robot_type, task='switch', frame_skip=frame_skip, time_step=0.02, action_robot_len=7, obs_robot_len=18)
		# self.observation_space = spaces.Box(-np.inf,np.inf,(18,), dtype=np.float32)
		self.observation_space = spaces.Box(-np.inf,np.inf,(15,), dtype=np.float32)
		self.success_dist = success_dist
		self.messages = np.array(['0 0 0','0 0 1','0 1 0', '0 1 1', '1 0 0', '1 0 1', '1 1 0', '1 1 1'])[message_indices]
		# self.num_targets = 2*len(self.messages)
		self.num_targets = 3
		self.switch_p = 1
		self.capture_frames = capture_frames

	def step(self, action):
		old_tool_pos = self.tool_pos

		self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))
		angle_dirs = np.zeros(len(self.switches))
		reward_switch = 0
		angle_diffs = []
		lever_angles = []
		for i,switch in enumerate(self.switches):
			angle_dirs[i],angle_diff = self.move_lever(switch)

			### Debugging: auto flip switch ###
			# tool_pos1 = np.array(p.getLinkState(self.tool, 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
			# if (norm(self.tool_pos-self.target_pos1[i]) < .07 or norm(tool_pos1-self.target_pos1[i]) < .1)\
			# 	or (norm(self.tool_pos-self.target_pos1[i]) < .07 or norm(tool_pos1-self.target_pos1[i]) < .1):
			# 	# for switch1 in self.switches:
			# 	if self.target_string[i] == 0:
			# 		p.resetJointState(switch, jointIndex=0, targetValue=LOW_LIMIT, physicsClientId=self.id)
			# 	else:
			# 		p.resetJointState(switch, jointIndex=0, targetValue=HIGH_LIMIT, physicsClientId=self.id)
			# 	self.update_targets()

			lever_angle = p.getJointStates(switch, jointIndices=[0], physicsClientId=self.id)[0][0]
			lever_angles.append(lever_angle)
			angle_diffs.append(angle_diff)
			if lever_angle < LOW_LIMIT + .1:
				self.current_string[i] = 0
			elif lever_angle > HIGH_LIMIT - .1:
				self.current_string[i] = 1
			else:
				self.current_string[i] = -1

			if self.target_string[i] == 0:
				reward_switch += -abs(LOW_LIMIT - lever_angle)
			else:
				reward_switch += -abs(HIGH_LIMIT - lever_angle)

			if self.target_string[i] == self.current_string[i]:
				p.changeVisualShape(self.targets1[i],-1,rgbaColor=[0,0,1,1])
				if self.target_string[i] == 0:
					p.resetJointState(switch, jointIndex=0, targetValue=LOW_LIMIT, physicsClientId=self.id)
				else:
					p.resetJointState(switch, jointIndex=0, targetValue=HIGH_LIMIT, physicsClientId=self.id)
				self.update_targets()
			else:
				p.changeVisualShape(self.targets1[i],-1,rgbaColor=[0,1,1,1])
		
		task_success = np.all(np.equal(self.current_string,self.target_string))
		self.task_success = task_success
		obs = self._get_obs([0])

		_,_,_, bad_contact_count = self.get_total_force()
		target_indices = np.nonzero(np.not_equal(self.target_string,self.current_string))[0]
		if len(target_indices) > 0:
			# reward_dist = -min([norm(self.tool_pos-self.target_pos[i]) for i in target_indices])
			reward_dist = -norm(self.tool_pos-self.target_pos[target_indices[0]])
		else:
			reward_dist = 0
		reward = 10*reward_dist + 10*reward_switch + (-30+10*np.count_nonzero(np.equal(self.target_string,self.current_string)))

		_,switch_orient = p.getBasePositionAndOrientation(self.wall, physicsClientId=self.id)
		info = {
			'task_success': self.task_success,
			'num_correct': np.count_nonzero(np.equal(self.target_string,self.current_string)),
			'angle_dir': angle_dirs,
			'angle_diff': angle_diffs,
			'tool_pos': self.tool_pos,
			'old_tool_pos': old_tool_pos,
			'ineff_contact': bad_contact_count,

			'target_index': self.target_index,
			'lever_angle': lever_angles,
			'target_string': self.target_string.copy(),
			'current_string': self.current_string.copy(),
			'switch_pos': np.array(self.target_pos).copy(),
			'aux_switch_pos': np.array(self.target_pos1).copy(),
			'switch_orient': switch_orient,
		}
		if self.capture_frames:
			frame = self.get_frame()
			info['frame'] = frame
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
		axis,_ = p.multiplyTransforms(np.zeros(3),p.getLinkState(switch,0)[1], [1,0,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
		centripedal = np.cross(axis,radius)
		c_F = np.dot(normal,centripedal)/norm(centripedal)
		k = .2
		w = k*np.sign(c_F)*np.sqrt(abs(c_F))*norm(radius)

		for _ in range(self.frame_skip):
			robot_joint_position += w
			
		robot_joint_position = np.clip(robot_joint_position,LOW_LIMIT,HIGH_LIMIT)
		p.resetJointState(switch, jointIndex=0, targetValue=robot_joint_position, physicsClientId=self.id)

		return w,robot_joint_position-old_j_pos

	def get_total_force(self):
		tool_force = 0
		tool_force_at_target = 0
		target_contact_pos = None
		bad_contact_count = 0
		for i in range(len(self.switches)):
			if self.target_string[i] == self.current_string[i]:
				for c in p.getContactPoints(bodyA=self.tool, bodyB=self.switches[i], physicsClientId=self.id):
					bad_contact_count += 1
		# for c in p.getContactPoints(bodyA=self.tool, bodyB=self.wall, physicsClientId=self.id):
		# 	bad_contact_count += 1
		return tool_force, tool_force_at_target, target_contact_pos, bad_contact_count

	def _get_obs(self, forces):
		torso_pos = np.array(p.getLinkState(self.robot, 15 if self.robot_type == 'pr2' else 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
		state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
		tool_pos = np.array(state[0])
		tool_orient = np.array(state[1]) # Quaternions
		robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
		robot_joint_positions = np.array([x[0] for x in robot_joint_states])
		robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

		# switch_pos = np.array(p.getBasePositionAndOrientation(self.switch, physicsClientId=self.id)[0])
		# robot_obs = np.concatenate([tool_pos-torso_pos, tool_orient, robot_joint_positions, switch_pos, forces]).ravel()

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

		"""set up target and initial robot position (objects set up with target)"""
		self.set_target_index() # instance override in demos
		self.generate_target()
		self.init_robot_arm()

		"""configure pybullet"""
		# p.setGravity(0, 0, -9.81, physicsClientId=self.id)
		p.setGravity(0, 0, 0, physicsClientId=self.id)
		p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
		# Enable rendering
		p.resetDebugVisualizerCamera(cameraDistance= .3, cameraYaw=180, cameraPitch=-10, cameraTargetPosition=[0, 0, 1.2], physicsClientId=self.id)
		p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
		self.viewMatrix = p.computeViewMatrixFromYawPitchRoll([0, 0, 1.2], .3, 180, -10, 0, 2)
		fov = 60
		aspect = (self.width/4)/(self.height/3)
		nearPlane = 0.01
		farPlane = 100
		self.projMatrix = p.computeProjectionMatrixFOV(fov,aspect,nearPlane,farPlane)

		return self._get_obs([0])
	
	def init_start_pos(self):
		"""exchange this function for curriculum"""
		init_pos = np.array([-0.2, -.5, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
		# switch_pos, switch_orient = p.getBasePositionAndOrientation(self.switches[0], physicsClientId=self.id)
		# self.init_pos, __ = p.multiplyTransforms(switch_pos, switch_orient, [0,.3,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)

	def init_robot_arm(self):
		self.init_start_pos()
		init_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)		
		self.util.ik_random_restarts(self.robot, 11, self.init_pos, init_orient, self.world_creation, self.robot_left_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits,
			ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=100, max_ik_random_restarts=10, random_restart_threshold=0.03, step_sim=True)
		# for joint_i,joint_pos in zip(self.robot_right_arm_joint_indices,[-.5, 2.8, 2, 2.4, -2.5, 1.2, 0.7]):
		# 	p.resetJointState(self.robot, jointIndex=joint_i, targetValue=joint_pos, physicsClientId=self.id)
		
		self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
		self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001]*3, pos_offset=[0, 0, 0.02], orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0], physicsClientId=self.id), maximal=False)

	def set_target_index(self):
		self.target_index = self.np_random.choice(self.num_targets)

	def generate_target(self): 
		# Place a switch on a wall
		# wall_index = self.target_index % 2
		wall_index = 0
		walls = [
			(np.array([0,-1.1,1]),[0,0,0,1]),
			(np.array([.65,-.4,1]),p.getQuaternionFromEuler([0, 0, np.pi/2])),
			# (np.array([-1.1,-.7,1]),p.getQuaternionFromEuler([0, 0, -np.pi/2])),
			]
		wall_pos,wall_orient = walls[wall_index]
		wall_collision = p.createCollisionShape(p.GEOM_BOX,halfExtents=[1,.1,1])
		wall_visual = p.createVisualShape(p.GEOM_BOX,halfExtents=[1,.1,1])
		self.wall = p.createMultiBody(basePosition=wall_pos,baseOrientation=wall_orient,baseCollisionShapeIndex=wall_collision,baseVisualShapeIndex=wall_visual,physicsClientId=self.id)
		# self.wall = p.createMultiBody(basePosition=wall_pos,baseOrientation=wall_orient,baseVisualShapeIndex=wall_visual,physicsClientId=self.id)

		self.target_string = np.array(self.messages[self.target_index].split(' ')).astype(int)
		# self.target_string = np.array(self.messages[self.target_index//2].split(' ')).astype(int)
		mask = self.np_random.choice([0,1],len(self.target_string),p=[1-self.switch_p,self.switch_p])
		if not np.count_nonzero(mask):
			mask = np.equal(np.arange(len(self.target_string)),self.np_random.choice(len(self.target_string))).astype(int)
		# self.initial_string = np.not_equal(self.target_string,mask).astype(int)
		self.initial_string = np.array([1,1,1])
		self.current_string = self.initial_string.copy()
		wall_pos, wall_orient = p.getBasePositionAndOrientation(self.wall, physicsClientId=self.id)
		switch_center = np.array([-.05-.15*(len(self.target_string)//2),.1,0])+np.array([.05,0,0])*self.np_random.uniform(-1,1,3)
		switch_scale = .075
		self.switches = []
		for increment,on_off in zip(np.linspace(np.zeros(3),[.15*(len(self.target_string)-1),0,0],num=len(self.target_string)),self.initial_string):
			switch_pos,switch_orient = p.multiplyTransforms(wall_pos, wall_orient, switch_center+increment, p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
			switch = p.loadURDF(os.path.join(self.world_creation.directory, 'light_switch', 'switch.urdf'),
								basePosition=switch_pos, useFixedBase=True, baseOrientation=switch_orient,\
								physicsClientId=self.id,globalScaling=switch_scale)
			self.switches.append(switch)
			p.setCollisionFilterPair(switch, switch, 0, -1, 0, physicsClientId=self.id)
			p.setCollisionFilterPair(switch, self.wall, 0, -1, 0, physicsClientId=self.id)
			p.setCollisionFilterPair(switch, self.wall, -1, -1, 0, physicsClientId=self.id)
			if not on_off:
				p.resetJointState(switch, jointIndex=0, targetValue=LOW_LIMIT+.099, physicsClientId=self.id)
			else:
				p.resetJointState(switch, jointIndex=0, targetValue=HIGH_LIMIT-.099, physicsClientId=self.id)

		sphere_collision = -1
		sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=self.success_dist, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
		self.targets = [p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[-10,-10,-10],
							useMaximalCoordinates=False, physicsClientId=self.id) for switch in self.switches]
		self.targets1 = [p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=[-10,-10,-10],
							useMaximalCoordinates=False, physicsClientId=self.id) for switch in self.switches]

		self.update_targets()

	def update_targets(self):
		self.target_pos = []
		self.target_pos1 = []
		for i,switch in enumerate(self.switches):
			switch_pos,switch_orient = p.getLinkState(switch, 0, computeForwardKinematics=True, physicsClientId=self.id)[:2]
			lever_pos = np.array([0,.07,.035])
			if self.target_string[i] == 0:
				second_pos = lever_pos + np.array([0,.03,.1])
				target_pos = np.array(p.multiplyTransforms(switch_pos, switch_orient, lever_pos, [0, 0, 0, 1])[0])
				target_pos1 = np.array(p.multiplyTransforms(switch_pos, switch_orient, second_pos, [0, 0, 0, 1])[0])
				self.target_pos.append(target_pos)
				self.target_pos1.append(target_pos1)
			else:
				second_pos = lever_pos + np.array([0,.03,-.1])
				target_pos = np.array(p.multiplyTransforms(switch_pos, switch_orient, lever_pos, [0, 0, 0, 1])[0])
				target_pos1 = np.array(p.multiplyTransforms(switch_pos, switch_orient, second_pos, [0, 0, 0, 1])[0])
				self.target_pos.append(target_pos)
				self.target_pos1.append(target_pos1)
				
			p.resetBasePositionAndOrientation(self.targets[i], target_pos, [0, 0, 0, 1], physicsClientId=self.id)
			p.resetBasePositionAndOrientation(self.targets1[i], target_pos1, [0, 0, 0, 1], physicsClientId=self.id)

	@property
	def tool_pos(self):
		return np.array(p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])

class OneSwitchJacoEnv(LightSwitchEnv):
	def __init__(self,**kwargs):
		super().__init__(message_indices=[3,5,6],robot_type='jaco',**kwargs)
class ThreeSwitchJacoEnv(LightSwitchEnv):
	def __init__(self,**kwargs):
		super().__init__(num_messages=[0,1,2],robot_type='jaco',**kwargs)
