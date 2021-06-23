from .asha_env import *
import os
from gym import spaces
import numpy as np
import pybullet as p
from numpy.linalg import norm
from gym.utils import seeding

LOW_LIMIT = -1
HIGH_LIMIT = .2


class LightSwitchEnv(AshaEnv):
    def __init__(self, num_targets=5,
                    success_dist=.03, session_goal=False, target_indices=None,
                    stochastic=True, robot_type='jaco', debug=False,
                    capture_frames=False, frame_skip=5):
        super().__init__(success_dist=success_dist, session_goal=session_goal, target_indices=target_indices,
                        stochastic=stochastic, robot_type=robot_type, debug=debug,
                        capture_frames=capture_frames, frame_skip=frame_skip)
        self.num_targets = 5

        self.goal_feat_sizes = {'original_switch_pos': 3}
        self.goal_set_shape = (self.num_targets, 4)

        self.wall_color = None
        self.scene_offset = None
        self.curr_step = 0
        self.step_limit = 200

        if target_indices is None:
            self.target_indices = list(np.arange(self.num_targets))
        else:
            for i in target_indices:
                assert 0 <= i < self.num_targets
            self.target_indices = target_indices

    def _step(self):
        angle_dirs = np.zeros(len(self.switches))
        reward_switch = 0
        angle_diffs = []
        self.lever_angles = []

        for i, switch in enumerate(self.switches):
            angle_dirs[i], angle_diff = self.move_lever(switch)

            ### Debugging: auto flip switch ###
            if self.debug:
                tool_pos1 = np.array(
                    p.getLinkState(self.tool, 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
                if (norm(self.tool_pos - self.target_pos1[i]) < .07 or norm(tool_pos1 - self.target_pos1[i]) < .1) \
                        or (
                        norm(self.tool_pos - self.target_pos1[i]) < .07 or norm(tool_pos1 - self.target_pos1[i]) < .1):
                    # for switch1 in self.switches:
                    if self.target_string[i] == 0:
                        p.resetJointState(switch, jointIndex=0, targetValue=LOW_LIMIT, physicsClientId=self.id)
                    else:
                        p.resetJointState(switch, jointIndex=0, targetValue=HIGH_LIMIT, physicsClientId=self.id)
                    self.update_targets()

            lever_angle = p.getJointStates(switch, jointIndices=[0], physicsClientId=self.id)[0][0]
            self.lever_angles.append(lever_angle)
            angle_diffs.append(angle_diff)
            if lever_angle < LOW_LIMIT + .1:
                self.current_string[i] = 0
            elif lever_angle > HIGH_LIMIT - .1:
                self.current_string[i] = 1
            else:
                self.current_string[i] = 1

            if self.target_string[i] == 0:
                reward_switch += -abs(LOW_LIMIT - lever_angle)
            else:
                reward_switch += -abs(HIGH_LIMIT - lever_angle)

            if self.target_string[i] == self.current_string[i]:
                self.update_targets()

        task_success = np.all(np.equal(self.current_string, self.target_string))
        self.task_success = task_success
        if self.task_success:
            color = [0, 1, 0, 1]
        elif self.wrong_goal_reached() or self.curr_step >= self.step_limit:
            color = [1, 0, 0, 1]
        else:
            color = [0, 0, 1, 1]
        p.changeVisualShape(self.targets1[self.target_index], -1, rgbaColor=color)

    def move_lever(self, switch):
        switch_pos, switch_orient = p.getLinkState(switch, 0)[:2]
        old_j_pos = robot_joint_position = p.getJointStates(switch, jointIndices=[0], physicsClientId=self.id)[0][0]
        contacts = p.getContactPoints(bodyA=self.robot, bodyB=switch, linkIndexB=0, physicsClientId=self.id)
        contacts += p.getContactPoints(bodyA=self.tool, bodyB=switch, linkIndexB=0, physicsClientId=self.id)
        if len(contacts) == 0:
            return 0, 0

        normal = contacts[0][7]
        k = -.01
        w = k * normal[2]

        for _ in range(self.frame_skip):
            robot_joint_position += w

        robot_joint_position = np.clip(robot_joint_position, LOW_LIMIT, HIGH_LIMIT)
        p.resetJointState(switch, jointIndex=0, targetValue=robot_joint_position, physicsClientId=self.id)

        return w, robot_joint_position - old_j_pos

    def get_total_force(self):
        tool_force = 0
        tool_force_at_target = 0
        target_contact_pos = None
        bad_contact_count = 0
        for i in range(len(self.switches)):
            if self.target_string[i] == self.current_string[i]:
                for c in p.getContactPoints(bodyA=self.tool, bodyB=self.switches[i], physicsClientId=self.id):
                    bad_contact_count += 1
        return tool_force, tool_force_at_target, target_contact_pos, bad_contact_count

    def _get_obs(self):
        obs = super()._get_obs()
        _, switch_orient = p.getBasePositionAndOrientation(self.wall, physicsClientId=self.id)

        obs.update({
            'lever_angle': self.lever_angles.copy(),
            'target_string': self.target_string.copy(),
            'current_string': self.current_string.copy(),
            'switch_pos': np.array(self.target_pos).copy(),
            'aux_switch_pos': np.array(self.target_pos1).copy(),
            'switch_orient': switch_orient,

            'num_correct': np.count_nonzero(np.equal(self.target_string, self.current_string)),
            'goal_set':np.concatenate((self.goal_positions, np.array(self.lever_angles)[:, None]),
                                    axis=1),
        })
        return obs

    def _reset(self):
        self.generate_target()

        self.camera_setting = dict(
            cameraDistance=.1,
            cameraYaw=180, cameraPitch=0,
            cameraTargetPosition=[0, -0.25, 1.3]
        )

        self.goal = self.target_pos[self.target_index].copy()

        self.lever_angles = [p.getJointStates(switch, jointIndices=[0], physicsClientId=self.id)[0][0]
                             for switch in self.switches]
        self.goal_positions = np.array(self.target_pos).copy()

    def init_start_pos(self):
        """exchange this function for curriculum"""
        self.init_pos = np.array([0, -.5, 1.1])
        if self.stochastic:
            self.init_pos += self.init_pos_random.uniform([-1]*3,[1]*3)*np.array([0.5, 0.1, 0.1])

    def get_random_target(self):
        targets = self.target_pos
        return targets[np.random.randint(self.num_targets)]

    def reset_noise(self):
        # default wall offset (for pretraining) is randomly chosen between (-0.1, 0)
        # calibration offset should be 0.1, online should be 0
        offset = self.np_random.choice((0.1, 0)) if self.scene_offset is None else self.scene_offset
        self.switch_pos_noise = [self.np_random.uniform(-.25, .05), 0, 0]
        self.wall_noise = [0, self.np_random.uniform(-.05, .05) + offset, 0]

    # return if a switch other than the target switch was flipped, assumes all switches start not flipped
    def wrong_goal_reached(self):
        return np.sum(self.current_string != self.target_string) > 1

    def generate_target(self):
        # Place a switch on a wall
        self.target_string = np.ones(self.num_targets).astype(int)
        self.target_string[self.target_index] = 0
        self.initial_string = np.ones(self.num_targets)
        self.current_string = self.initial_string.copy()
        wall_pos, wall_orient = p.getBasePositionAndOrientation(self.wall, physicsClientId=self.id)

        switch_spacing = .88 / (self.num_targets - 1)
        switch_center = np.array([-switch_spacing * (len(self.target_string) // 2), .1, 0])
        if self.stochastic:
            switch_center = switch_center + self.switch_pos_noise
        switch_scale = .075
        self.switches = []
        for increment, on_off in zip(np.linspace(np.zeros(3), [switch_spacing * (len(self.target_string) - 1), 0, 0],
                                                 num=len(self.target_string)), self.initial_string):
            switch_pos, switch_orient = p.multiplyTransforms(wall_pos, wall_orient, switch_center + increment,
                                                             p.getQuaternionFromEuler([0, 0, 0]),
                                                             physicsClientId=self.id)
            switch = p.loadURDF(os.path.join(self.world_creation.directory, 'light_switch', 'switch.urdf'),
                                basePosition=switch_pos, useFixedBase=True, baseOrientation=switch_orient, \
                                physicsClientId=self.id, globalScaling=switch_scale)
            self.switches.append(switch)
            p.setCollisionFilterPair(switch, switch, 0, -1, 0, physicsClientId=self.id)
            p.setCollisionFilterPair(switch, self.wall, 0, -1, 0, physicsClientId=self.id)
            p.setCollisionFilterPair(switch, self.wall, -1, -1, 0, physicsClientId=self.id)
            if not on_off:
                p.resetJointState(switch, jointIndex=0, targetValue=LOW_LIMIT, physicsClientId=self.id)
            else:
                p.resetJointState(switch, jointIndex=0, targetValue=HIGH_LIMIT, physicsClientId=self.id)

        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=self.success_dist * 1.5,
                                            rgbaColor=[0, 0, 1, 1], physicsClientId=self.id)

        self.targets = []
        self.targets1 = []
        for i, switch in enumerate(self.switches):
            self.targets.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                                  basePosition=[-10, -10, -10], useMaximalCoordinates=False,
                                                  physicsClientId=self.id))
            if i == self.target_index:
                self.targets1.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                                       baseVisualShapeIndex=sphere_visual, basePosition=[-10, -10, -10],
                                                       useMaximalCoordinates=False, physicsClientId=self.id))
            else:
                self.targets1.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                                       basePosition=[-10, -10, -10],
                                                       useMaximalCoordinates=False, physicsClientId=self.id))

        self.update_targets()

    def update_targets(self):
        self.target_pos = []
        self.target_pos1 = []
        for i, switch in enumerate(self.switches):
            switch_pos, switch_orient = p.getLinkState(switch, 0, computeForwardKinematics=True,
                                                       physicsClientId=self.id)[:2]
            lever_pos = np.array([0, .07, .035])
            if self.target_string[i] == 0:
                second_pos = lever_pos + np.array([0, .03, .1])
                target_pos = np.array(p.multiplyTransforms(switch_pos, switch_orient, lever_pos, [0, 0, 0, 1])[0])
                target_pos1 = np.array(p.multiplyTransforms(switch_pos, switch_orient, second_pos, [0, 0, 0, 1])[0])
                self.target_pos.append(target_pos)
                self.target_pos1.append(target_pos1)
            else:
                second_pos = lever_pos + np.array([0, .03, -.1])
                target_pos = np.array(p.multiplyTransforms(switch_pos, switch_orient, lever_pos, [0, 0, 0, 1])[0])
                target_pos1 = np.array(p.multiplyTransforms(switch_pos, switch_orient, second_pos, [0, 0, 0, 1])[0])
                self.target_pos.append(target_pos)
                self.target_pos1.append(target_pos1)

            p.resetBasePositionAndOrientation(self.targets[i], target_pos, [0, 0, 0, 1], physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.targets1[i], target_pos1, [0, 0, 0, 1], physicsClientId=self.id)


class OneSwitchJacoEnv(LightSwitchEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_type='jaco', **kwargs)
