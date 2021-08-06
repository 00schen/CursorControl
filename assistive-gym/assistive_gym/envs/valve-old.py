# Environment is not present in original assistive_gym library at https://github.com/Healthcare-Robotics/assistive-gym

from gym import spaces
import numpy as np
import pybullet as p
from .env import AssistiveEnv
from gym.utils import seeding
from collections import OrderedDict
import os

reach_arena = (np.array([-.25, -.5, 1]), np.array([.6, .4, .2]))
default_orientation = p.getQuaternionFromEuler([0, 0, 0])


class ValveEnv(AssistiveEnv):
    def __init__(self, robot_type='jaco', success_dist=.05, target_indices=None, session_goal=False, frame_skip=5,
                 capture_frames=False, stochastic=True, debug=False, step_limit=200, min_error_threshold=np.pi / 16,
                 max_error_threshold=np.pi / 4, success_threshold=20, num_targets=None, use_rand_init_angle=True):
        super(ValveEnv, self).__init__(robot_type=robot_type, task='reaching', frame_skip=frame_skip, time_step=0.02,
                                       action_robot_len=7, obs_robot_len=14)
        self.observation_space = spaces.Box(-np.inf, np.inf, (7 + 3 + 2 + 7 + 1,), dtype=np.float32)
        self.num_targets = num_targets
        self.success_dist = success_dist
        self.debug = debug
        self.stochastic = stochastic
        self.goal_feat = ['target_angle']  # Just an FYI
        self.feature_sizes = OrderedDict({'goal': 2})
        self.session_goal = session_goal
        self.use_rand_init_angle = use_rand_init_angle

        if self.num_targets is not None:
            self.target_indices = list(np.arange(self.num_targets))
            self.target_angles = np.linspace(-np.pi, np.pi, self.num_targets, endpoint=False)

        self.min_error_threshold = min_error_threshold
        self.max_error_threshold = max_error_threshold
        self.error_threshold = min_error_threshold
        self.success_threshold = success_threshold

        self.wall_color = None
        self.step_limit = step_limit
        self.curr_step = 0
        self.last_angle = None
        self.last_moved = 0
        self.has_moved = False
        self.calibrate = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.init_pos_random, _ = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.curr_step += 1
        old_tool_pos = self.tool_pos

        self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

        obs = self._get_obs([0])

        if self.last_angle == self.valve_angle:
            self.last_moved += 1
        else:
            self.last_moved = 0
            self.has_moved = True
        self.last_angle = self.valve_angle

        self.task_success = self.is_success and self.last_moved >= self.success_threshold

        if self.task_success:
            color = [0, 1, 0, 1]
        elif self.is_success:
            color = [0, 0.5, 0, 1]
        elif self.wrong_goal_reached() or self.curr_step >= self.step_limit:
            color = [1, 0, 0, 1]
        else:
            color = [0, 0, 1, 1]
        p.changeVisualShape(self.target_indicator, -1, rgbaColor=color)

        # if self.task_success:
        #     target_color = [0, 1, 0, 1]
        #     p.changeVisualShape(self.target, -1, rgbaColor=target_color)
        # elif self.curr_step >= self.step_limit:
        #     target_color = [1, 0, 0, 1]
        #     p.changeVisualShape(self.target, -1, rgbaColor=target_color)

        if self.task_success:
            reward = 0
        else:
            reward = -np.abs(self.angle_diff(self.valve_angle, self.target_angle)) / np.pi

            # reward so far between (-1, 0), want to make between (-0.5, -0.1)
            # reward = reward / 2
            # reward = 0.4 * reward - 0.1
            reward = 0.9 * reward - 0.1
            # if not (self.is_success or self.has_moved):
            #     reward += (np.exp(-np.linalg.norm(self.valve_pos - self.tool_pos)) - 1) / 2

        direction = np.zeros(3)
        if self.is_success:
            index = 0
        else:
            index = 1 if self.angle_diff(self.valve_angle, self.target_angle) > 0 else 2
        direction[index] = 1

        info = {
            'task_success': self.task_success,
            'old_tool_pos': old_tool_pos,
            'tool_pos': self.tool_pos,
            'valve_pos': self.valve_pos,
            'valve_angle': [self.valve_angle],
            'target_angle': [self.target_angle],
            'error_threshold': [self.error_threshold],
            'direction': direction,
            'is_success': self.is_success
        }
        done = self.task_success
        return obs, reward, done, info

    def _get_obs(self, forces):
        robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices,
                                              physicsClientId=self.id)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])

        robot_obs = dict(
            raw_obs=np.concatenate([self.tool_pos, self.tool_orient, self.valve_pos,
                                    [np.sin(self.valve_angle), np.cos(self.valve_angle)], robot_joint_positions,
                                    [min(self.last_moved, self.success_threshold) / self.success_threshold,]]),
            hindsight_goal=np.array([np.sin(self.valve_angle), np.cos(self.valve_angle)]),
            goal=self.goal.copy(),
        )
        return robot_obs

    def update_curriculum(self, success):
        if success:
            self.error_threshold -= self.min_error_threshold
            self.error_threshold = max(self.min_error_threshold, self.error_threshold)
        else:
            self.error_threshold += self.min_error_threshold
            self.error_threshold = min(self.max_error_threshold, self.error_threshold)

    def reset(self):
        self.task_success = False

        """set up standard environment"""
        self.setup_timing()
        _human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, _human_lower_limits, \
        _human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender \
            = self.world_creation.create_new_world(furniture_type='wheelchair', init_human=False,
                                                   static_human_base=True, human_impairment='random',
                                                   print_joints=False, gender='random')
        self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
        self.reset_robot_joints()
        wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
        p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]),
                                          p.getQuaternionFromEuler([0, 0, -np.pi / 2.0], physicsClientId=self.id),
                                          physicsClientId=self.id)
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
        self.human_controllable_joint_indices = []
        self.human_lower_limits = np.array([])
        self.human_upper_limits = np.array([])

        """set up target and initial robot position"""
        if not self.session_goal:
            self.set_target_index()  # instance override in demos
            self.reset_noise()

        self.init_robot_arm()

        wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, .1, 1])
        wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[4, .1, 1], rgbaColor=self.wall_color)

        wall_pos, wall_orient = np.array([0., -1.1, 1.]), np.array([0, 0, 0, 1])
        if self.stochastic:
            wall_pos = wall_pos + + self.wall_noise

        self.wall = p.createMultiBody(basePosition=wall_pos, baseOrientation=wall_orient,
                                      baseCollisionShapeIndex=wall_collision, baseVisualShapeIndex=wall_visual,
                                      physicsClientId=self.id)

        valve_pos, valve_orient = p.multiplyTransforms(wall_pos, wall_orient, [0, 0.1, 0],
                                                       p.getQuaternionFromEuler([0, 0, 0]),
                                                       physicsClientId=self.id)
        if self.stochastic:
            valve_pos = np.array(valve_pos) + self.valve_pos_noise

        self.valve = p.loadURDF(os.path.join(self.world_creation.directory, 'valve', 'valve.urdf'),
                                basePosition=valve_pos, useFixedBase=True,
                                baseOrientation=valve_orient, globalScaling=1,
                                physicsClientId=self.id)

        if self.use_rand_init_angle:
            p.resetJointState(self.valve, 0, self.rand_init_angle, physicsClientId=self.id)

        """configure pybullet"""
        p.setGravity(0, 0, 0, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
        # Enable rendering
        p.resetDebugVisualizerCamera(cameraDistance=.1, cameraYaw=180, cameraPitch=-10,
                                     cameraTargetPosition=[0, 0, 1.1], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.task_success = 0
        self.goal = np.array([np.sin(self.target_angle), np.cos(self.target_angle)])
        self.curr_step = 0

        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05,
                                            rgbaColor=[0, 0, 1, 1], physicsClientId=self.id)

        target_coord = 0.4 * np.array((-np.cos(self.target_angle), 0, np.sin(self.target_angle))) + valve_pos + \
                       [0, 0.105, 0]

        self.target_indicator = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                                  baseVisualShapeIndex=sphere_visual, basePosition=target_coord,
                                                  useMaximalCoordinates=False, physicsClientId=self.id)

        self.last_angle = self.valve_angle
        self.last_moved = 0
        self.has_moved = False
        return self._get_obs([0])

    def init_start_pos(self):
        """exchange this function for curriculum"""
        self.init_pos = np.array([0, -.5, 1.1])

        # if self.stochastic:
        self.init_pos += self.init_pos_random.uniform([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1], size=3)

    def init_robot_arm(self):
        self.init_start_pos()
        init_orient = p.getQuaternionFromEuler(np.array([0, np.pi / 2.0, 0]), physicsClientId=self.id)
        self.util.ik_random_restarts(self.robot, 11, self.init_pos, init_orient, self.world_creation,
                                     self.robot_left_arm_joint_indices, self.robot_lower_limits,
                                     self.robot_upper_limits,
                                     ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=100,
                                     max_ik_random_restarts=10, random_restart_threshold=0.03, step_sim=True)
        self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
        self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001] * 3, pos_offset=[0, 0, 0.02],
                                                  orient_offset=p.getQuaternionFromEuler([0, -np.pi / 2.0, 0],
                                                                                         physicsClientId=self.id),
                                                  maximal=False)

    def set_target_index(self, index=None):
        if self.num_targets is not None:
            if index is None:
                self.target_index = self.np_random.choice(self.target_indices)
            else:
                self.target_index = index

    def reset_noise(self):

        self.rand_init_angle = (np.random.rand() - 0.5) * 2 * np.pi

        # init angle either self.rand_init_angle or 0
        avoid = self.rand_init_angle if self.use_rand_init_angle else 0

        self.rand_angle = None
        while self.rand_angle is None or np.abs(self.angle_diff(self.rand_angle, avoid)) < self.error_threshold:
            self.rand_angle = (np.random.rand() - 0.5) * 2 * np.pi

        self.valve_pos_noise = np.array([self.np_random.uniform(-.1, .1), 0, 0])
        self.wall_noise = np.array([0, self.np_random.uniform(-.1, .1), 0])

    def wrong_goal_reached(self):
        return self.has_moved and self.last_moved >= self.success_threshold and not self.is_success

    def calibrate_mode(self, calibrate, split):
        self.wall_color = [255 / 255, 187 / 255, 120 / 255, 1] if calibrate else None
        self.calibrate = calibrate

    @property
    def tool_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[0])

    @property
    def tool_orient(self):
        return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[1])

    @property
    def valve_pos(self):
        return p.getLinkState(self.valve, 0, computeForwardKinematics=True, physicsClientId=self.id)[0]

    @property
    def valve_angle(self):
        return self.wrap_angle(p.getJointStates(self.valve, jointIndices=[0], physicsClientId=self.id)[0][0])

    @property
    def target_angle(self):
        return self.rand_angle if self.num_targets is None or not self.calibrate else \
            self.wrap_angle(self.target_angles[self.target_index])

    def wrap_angle(self, angle):
        return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))

    def angle_diff(self, angle1, angle2):
        a = angle1 - angle2
        if a > np.pi:
            a -= 2 * np.pi
        elif a < -np.pi:
            a += 2 * np.pi
        return a

    @property
    def is_success(self):
        return np.abs(self.angle_diff(self.valve_angle, self.target_angle)) < self.error_threshold


class ValveJacoEnv(ValveEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_type='jaco', **kwargs)
