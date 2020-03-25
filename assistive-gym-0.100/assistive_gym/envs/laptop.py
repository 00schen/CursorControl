import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv

class LaptopEnv(AssistiveEnv):
    def __init__(self, robot_type='Jaco', human_control=False):
        super(LaptopEnv, self).__init__(robot_type=robot_type, task='laptop', human_control=human_control,\
            frame_skip=10, time_step=0.01, action_robot_len=7, action_human_len=(10 if human_control else 0),\
            obs_robot_len=30, obs_human_len=(34 if human_control else 0))

    def step(self, action):
        self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

        hand_force, hand_force_at_target, target_contact_pos = self.get_total_force()
        obs = self._get_obs([hand_force], [hand_force_at_target])

        # TODO: look at which link hand is
        self.hand_pos = np.array(p.getLinkState(self.robot, 7, computeForwardKinematics=True, physicsClientId=self.id)[0])
        reward_distance = -np.linalg.norm(self.target_pos - self.hand_pos) # Penalize distances away from target
        reward_action = -np.sum(np.square(action)) # Penalize actions
        reward_force_target = 0.0 # Reward force at the target
        laptop_angle = self._get_laptop_angle()
        if target_contact_pos is not None\
            and np.linalg.norm(target_contact_pos - self.prev_target_contact_pos) > 0.01\
            and hand_force_at_target < 10\
            and laptop_angle > np.pi/2 and laptop_angle < np.deg2rad(110):
            self.task_success += 1

        reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('hand_force_weight')*hand_force_at_target

        if self.gui and hand_force_at_target > 0:
            print('Task success:', self.task_success, 'hand force at target:', hand_force_at_target)

        info = {'task_success': int(self.task_success >= self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = False

        return obs, reward, done, info

    def get_total_force(self):
        hand_force = 0
        hand_force_at_target = 0
        target_contact_pos = None
        for c in p.getContactPoints(bodyA=self.robot, physicsClientId=self.id):
            hand_force += c[9]
        for c in p.getContactPoints(bodyA=self.robot, bodyB=self.laptop, physicsClientId=self.id):
            linkA = c[3]
            contact_position = c[6]
            if linkA in [6, 7]: # TODO: Figure out which links are fingers
                # Enforce that contact is close to the target location
                if np.linalg.norm(contact_position - self.target_pos) < 0.025:
                    hand_force_at_target += c[9]
                    target_contact_pos = np.array(contact_position)
        return hand_force, hand_force_at_target, target_contact_pos

    def _get_obs(self, forces, forces_human):
        torso_pos = np.array(p.getLinkState(self.robot, 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        state = p.getLinkState(self.robot, 7, computeForwardKinematics=True, physicsClientId=self.id) # TODO: figure out which link is hand
        hand_pos = np.array(state[0])
        hand_orient = np.array(state[1]) # Quaternions
        robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices, physicsClientId=self.id)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])
        robot_pos, robot_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
        lbody_pos = np.array(p.getLinkState(self.laptop, 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        screen_pos = np.array(p.getLinkState(self.laptop, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])

        robot_obs = np.concatenate([hand_pos-torso_pos, hand_orient, hand_pos - self.target_pos, self.target_pos-torso_pos, robot_joint_positions, lbody_pos-torso_pos, screen_pos-torso_pos, forces]).ravel()

        return robot_obs

    def _get_laptop_angle(self):
        joint_pos = np.array(p.getJointStates(self.laptop,0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        lbody_pos = np.array(p.getLinkState(self.laptop, 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        screen_pos = np.array(p.getLinkState(self.laptop, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])

        return np.arctan2(screen_pos-joint_pos,lbody_pos-joint_pos)


    def reset(self):
        self.setup_timing()
        self.task_success = 0
        self.human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, self.human_lower_limits, self.human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender = self.world_creation.create_new_world(furniture_type='wheelchair', static_human_base=True, human_impairment='random', print_joints=False, gender='random')
        self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
        self.reset_robot_joints()
        if self.robot_type == 'jaco':
            wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]), p.getQuaternionFromEuler([0, 0, -np.pi/2.0], physicsClientId=self.id), physicsClientId=self.id)
            base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)

        joints_positions = [(6, np.deg2rad(-90)), (16, np.deg2rad(-90)), (28, np.deg2rad(-90)), (31, np.deg2rad(80)), (35, np.deg2rad(-90)), (38, np.deg2rad(80))]
        joints_positions += [(21, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (22, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30))), (23, self.np_random.uniform(np.deg2rad(-30), np.deg2rad(30)))]
        self.human_controllable_joint_indices = [20, 21, 22, 23] # TODO: change to arm
        self.world_creation.setup_human_joints(self.human, joints_positions, self.human_controllable_joint_indices if (self.human_control or self.world_creation.human_impairment == 'tremor') else [], use_static_joints=True, human_reactive_force=None)
        p.resetBasePositionAndOrientation(self.human, [0, 0.03, 0.89 if self.gender == 'male' else 0.86], [0, 0, 0, 1], physicsClientId=self.id)
        human_joint_states = p.getJointStates(self.human, jointIndices=self.human_controllable_joint_indices, physicsClientId=self.id)
        self.target_human_joint_positions = np.array([x[0] for x in human_joint_states])
        self.human_lower_limits = self.human_lower_limits[self.human_controllable_joint_indices]
        self.human_upper_limits = self.human_upper_limits[self.human_controllable_joint_indices]

        # Place a laptop of food on a table
        self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'), basePosition=[0.35, -0.9, 0], baseOrientation=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)
        self.laptop_scale = 0.75
        # TODO: Figure out dimensions
        laptop_pos = np.array([-0.15, -0.55, 0.75]) + np.array([self.np_random.uniform(-0.05, 0.05), self.np_random.uniform(-0.05, 0.05), 0])
        self.laptop = p.loadURDF(os.path.join(self.world_creation.directory, 'laptop', 'laptop.urdf'), basePosition=laptop_pos, baseOrientation=p.getQuaternionFromEuler([0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)

        shoulder_pos, shoulder_orient = p.getLinkState(self.human, 5, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        elbow_pos, elbow_orient = p.getLinkState(self.human, 7, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        wrist_pos, wrist_orient = p.getLinkState(self.human, 9, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        head_pos, head_orient = p.getLinkState(self.human, 23, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        if self.robot_type == 'pr2':
            target_orient = p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id)
            self.position_robot_toc(self.robot, 54, [(target_pos, target_orient), (self.target_pos, None)], [(self.target_pos, target_orient)], self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(15, 15+7), pos_offset=np.array([0.1, 0.2, 0]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.world_creation.set_gripper_open_position(self.robot, position=0.03, left=False, set_instantly=True)
        elif self.robot_type == 'jaco':
            target_orient = p.getQuaternionFromEuler(np.array([np.pi/2.0, 0, np.pi/2.0]), physicsClientId=self.id)
            self.util.ik_random_restarts(self.robot, 8, target_pos, target_orient, self.world_creation, self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=1000, max_ik_random_restarts=40, random_restart_threshold=0.01, step_sim=True, check_env_collisions=True)
            self.world_creation.set_gripper_open_position(self.robot, position=1.33, left=False, set_instantly=True)
        else:
            target_orient = p.getQuaternionFromEuler(np.array([np.pi/2.0, 0, np.pi/2.0]), physicsClientId=self.id)
            if self.robot_type == 'baxter':
                self.position_robot_toc(self.robot, 26, [(target_pos, target_orient)], [(self.target_pos, target_orient)], self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=range(1, 8), pos_offset=np.array([0, 0.2, 0.975]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            else:
                self.position_robot_toc(self.robot, 19, [(target_pos, target_orient), (self.target_pos, None)], [(self.target_pos, target_orient)], self.robot_right_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, ik_indices=[0, 2, 3, 4, 5, 6, 7], pos_offset=np.array([-0.1, 0.2, 0.975]), max_ik_iterations=200, step_sim=True, check_env_collisions=False, human_joint_indices=self.human_controllable_joint_indices, human_joint_positions=self.target_human_joint_positions)
            self.world_creation.set_gripper_open_position(self.robot, position=0.0, left=False, set_instantly=True)

        p.resetBasePositionAndOrientation(self.laptop, laptop_pos, p.getQuaternionFromEuler([np.pi/2.0, 0, 0], physicsClientId=self.id), physicsClientId=self.id)

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)
        p.setGravity(0, 0, 0, body=self.human, physicsClientId=self.id)

        p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return self._get_obs([0], [0, 0])

    def generate_target(self): 
        lbody_pos, lbody_orient = p.getLinkState(self.laptop, 0, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        self.target_button = np.array([-0.002,-0.001,0.0025]) + np.concatenate((1e-4*np.random.random(2),[0])) # TODO: figure out dimensions
        target_pos, target_orient = p.multiplyTransforms(lbody_pos, lbody_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)

        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.id)
        self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

        self.update_targets()

    def update_targets(self):
        lbody_pos, lbody_orient = p.getLinkState(self.laptop, 0, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        target_pos, target_orient = p.multiplyTransforms(lbody_pos, lbody_orient, self.target_on_arm, [0, 0, 0, 1], physicsClientId=self.id)
        self.target_pos = np.array(target_pos)
        p.resetBasePositionAndOrientation(self.target, self.target_pos, [0, 0, 0, 1], physicsClientId=self.id)
