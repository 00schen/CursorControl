from .asha_env import *
import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from gym.utils import seeding

reach_arena = (np.array([-.25, -.5, 1]), np.array([.6, .4, .2]))
default_orientation = p.getQuaternionFromEuler([0, 0, 0])


class BottleEnv(AshaEnv):
    def __init__(self,
                    success_dist=.03, session_goal=False, target_indices=None,
                    stochastic=True, robot_type='jaco', debug=False,
                    capture_frames=False, frame_skip=5):
        super().__init__(success_dist=success_dist, session_goal=session_goal, target_indices=target_indices,
                        stochastic=stochastic, robot_type=robot_type, debug=debug,
                        capture_frames=capture_frames, frame_skip=frame_skip)
        self.num_targets = 4

        self.goal_feat_sizes = {'target_pos': 3}
        self.goal_set_shape = (2, 3)

        self.wall_color = None
        self.scene_offset = None
        self.curr_step = 0

    def _step(self):
        self.move_slide_cabinet()

        self.task_success = norm(self.tool_pos - self.target_pos) < self.success_dist

        if self.task_success:
            target_color = [0, 1, 0, 1]
            p.changeVisualShape(self.target, -1, rgbaColor=target_color)
        elif self.wrong_goal_reached() or self.curr_step >= self.step_limit:
            target_color = [1, 0, 0, 1]
            p.changeVisualShape(self.target, -1, rgbaColor=target_color)


    def move_slide_cabinet(self):
        robot_joint_position = p.getJointStates(self.shelf, jointIndices=[0], physicsClientId=self.id)[0][0]
        contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.shelf, linkIndexB=1, physicsClientId=self.id)
        contacts += p.getContactPoints(bodyA=self.tool, bodyB=self.shelf, linkIndexB=1, physicsClientId=self.id)
        contacts += p.getContactPoints(bodyA=self.robot, bodyB=self.shelf, linkIndexB=0, physicsClientId=self.id)
        contacts += p.getContactPoints(bodyA=self.tool, bodyB=self.shelf, linkIndexB=0, physicsClientId=self.id)
        if len(contacts) == 0 or norm(self.tool_pos - self.door_pos) > .1:
            return 0, 0

        normal = contacts[0][7]
        c_F = -normal[0]
        k = .002
        w = k * np.sign(c_F) * np.sqrt(abs(c_F))
        for _ in range(self.frame_skip):
            robot_joint_position += w
        robot_joint_position = np.clip(robot_joint_position, *self.slide_range)
        p.resetJointState(self.shelf, jointIndex=0, targetValue=robot_joint_position, physicsClientId=self.id)

    def _get_obs(self):
        obs = super()._get_obs()

        obs.update({
            'door_open': self.door_open,
            'door_pos': self.door_pos,
            'shelf_pos': self.shelf_pos,
            'sub_target': self.sub_target_pos.copy(),
            'target_pos': self.target_pos,
            'goal_set': self.bottle_poses,
        })
        return obs

    def _reset(self):
        """set up shelf environment objects"""
        self.table_pos = table_pos = np.array([0, -1.05, 0])
        if self.stochastic:
            self.table_pos = table_pos = table_pos + self.table_noise

        self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'),
                                basePosition=table_pos, baseOrientation=default_orientation, physicsClientId=self.id)

        self.generate_target()

        self.camera_setting = dict(
            cameraDistance=.1,
            cameraYaw=180, cameraPitch=-30,
            cameraTargetPosition=[0, -.25, 1.0]
        )

        self.goal = np.array(self.target_pos)

    def init_start_pos(self):
        """exchange this function for curriculum"""
        self.init_pos = np.array([0, -.5, 1.1])
        if self.stochastic:
            self.init_pos += self.init_pos_random.uniform([-1]*3,[1]*3)*np.array([0.4, 0.1, 0.1])

    def get_random_target(self):
        return self.bottle_poses[np.random.randint(2)]

    def reset_noise(self):
        offset = self.np_random.choice((0.1, 0)) if self.scene_offset is None else self.scene_offset
        self.table_noise = self.np_random.uniform([-0.25, -0.05, 0], [0.15, 0.05, 0], size=3)
        self.table_noise[1] = self.table_noise[1] + offset
        self.wall_noise = 0

    def wrong_goal_reached(self):
        return norm(self.tool_pos - self.wrong_target_pos) < self.success_dist

    def generate_target(self):
        self.shelf_pos = self.table_pos + np.array([0, .1, 1])
        if self.target_index % 2:
            cabinet_urdf, self.slide_range = 'slide_cabinet_left.urdf', (-.3, 0)
        else:
            cabinet_urdf, self.slide_range = 'slide_cabinet_right.urdf', (0, .3)
        self.shelf = p.loadURDF(os.path.join(self.world_creation.directory, 'slide_cabinet', cabinet_urdf),
                                basePosition=self.shelf_pos, baseOrientation=default_orientation, globalScaling=1,
                                physicsClientId=self.id, useFixedBase=True)

        bottle_poses = []
        for increment in product(np.linspace(-.15, .15, num=2), [0], [-.15]):
            bottle_pos, bottle_orient = p.multiplyTransforms(self.shelf_pos, default_orientation, increment,
                                                             default_orientation, physicsClientId=self.id)
            bottle_poses.append(bottle_pos + np.array([0, 0, .05]))
        self.bottle_poses = np.array(bottle_poses)

        target_pos = self.target_pos = bottle_poses[self.target_index // 2]
        wrong_target_pos = self.wrong_target_pos = bottle_poses[1 - (self.target_index // 2)]

        bottle_pos = target_pos + np.array([0, 0, -.05])
        self.bottle = p.loadURDF(os.path.join(self.world_creation.directory, 'bottle', 'bottle.urdf'),
                                 basePosition=bottle_pos, useFixedBase=True,
                                 baseOrientation=default_orientation, globalScaling=.01,
                                 physicsClientId=self.id)

        wrong_bottle_pos = wrong_target_pos + np.array([0, 0, -.05])
        self.wrong_bottle = p.loadURDF(
            os.path.join(self.world_creation.directory, 'bottle', 'bottle.urdf'),
            basePosition=wrong_bottle_pos, useFixedBase=True,
            baseOrientation=default_orientation, globalScaling=.01,
            physicsClientId=self.id)

        sphere_collision = -1

        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1],
                                            physicsClientId=self.id)
        self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                        baseVisualShapeIndex=sphere_visual, basePosition=target_pos,
                                        useMaximalCoordinates=False, physicsClientId=self.id)
        self.slide = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                       basePosition=self.door_pos,
                                       useMaximalCoordinates=False, physicsClientId=self.id)
        self.update_targets()

    def update_targets(self):
        p.resetBasePositionAndOrientation(self.slide, self.door_pos, [0, 0, 0, 1], physicsClientId=self.id)

    @property
    def door_pos(self):
        location = p.getLinkState(self.shelf, 1, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        door_pos = [.15, .17, 0] if self.target_index % 2 else [-.15, .17, 0]
        return np.array(p.multiplyTransforms(*location, door_pos, [0, 0, 0, 1])[0])

    @property
    def door_open(self):
        final_door_pos = (np.array([-.15, .17, 0]) if self.target_index // 2 else np.array(
            [.15, .17, 0])) + self.shelf_pos
        return norm(self.door_pos - final_door_pos) < .02

    @property
    def sub_target_pos(self):
        final_door_pos = (np.array([-.15, .17, 0]) if self.target_index // 2 else np.array(
            [.15, .17, 0])) + self.shelf_pos
        return self.target_pos if self.door_open else final_door_pos


class BottleJacoEnv(BottleEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_type='jaco', **kwargs)
