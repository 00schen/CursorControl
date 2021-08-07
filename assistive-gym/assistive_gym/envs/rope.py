# Environment is not present in original assistive_gym library at https://github.com/Healthcare-Robotics/assistive-gym

import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from .env import AssistiveEnv
from gym.utils import seeding
from collections import OrderedDict

reach_arena = (np.array([-.25, -.5, 1]), np.array([.6, .4, .2]))
default_orientation = p.getQuaternionFromEuler([0, 0, 0])
tau = 2 * np.pi

class RopeEnv(AssistiveEnv):
    def __init__(self, robot_type='jaco', success_dist=.05,session_goal=False, frame_skip=5,
                 capture_frames=False, stochastic=True, debug=False, step_limit=200, **kwargs):
        super().__init__(robot_type=robot_type, task='switch', frame_skip=frame_skip, time_step=0.02,
                                        action_robot_len=7, obs_robot_len=14)
        self.num_links = 100
        self.observation_space = spaces.Box(-np.inf, np.inf, (7 + self.num_links*3 + 7,), dtype=np.float32)  # TODO: observation space size
        self.num_targets = 2
        self.success_dist = success_dist
        self.debug = debug
        self.stochastic = stochastic
        self.feature_sizes = OrderedDict({'goal': 6})  # TODO: rope goal features
        self.session_goal = session_goal
        self.target_indices = list(np.arange(self.num_targets))
        self.table_offset = None

        self.wall_color = None
        self.step_limit = step_limit
        self.curr_step = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.init_pos_random, _ = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.curr_step += 1
        old_tool_pos = self.tool_pos

        joint_action = action[:7]
        self.grab = action[-1]
        if self.grab and not self.grab_constraints:
            self.grab_constraints = []
            for link in self.rope:
                link_pos = p.getBasePositionAndOrientation(link)[0]
                if p.getContactPoints(bodyA=self.tool, bodyB=link):
                    constraint_id = p.createConstraint(
                            parentBodyUniqueId=self.tool,
                            parentLinkIndex=-1,
                            childBodyUniqueId=link,
                            childLinkIndex=-1,
                            jointType=p.JOINT_POINT2POINT,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=(0, 0, 0),
                            childFramePosition=(0, 0, 0))
                    p.changeConstraint(constraint_id, maxForce=int(1e6))
                    self.grab_constraints.append(constraint_id)
        if not self.grab:
            for constraint in self.grab_constraints:
                p.removeConstraint(constraint)
            self.grab_constraints = []

        self.take_step(joint_action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))

        obs = self._get_obs([0])

        rope_pos = np.concatenate([
            p.getBasePositionAndOrientation(link)[0] for link in self.rope
        ])
        frechet = norm(rope_pos-np.concatenate(self.goal_points))
        self.task_success = frechet < .15

        # if self.task_success:
        #     target_color = [0, 1, 0, 1]
        #     p.changeVisualShape(self.target, -1, rgbaColor=target_color)
        # elif self.curr_step >= self.step_limit:
        #     target_color = [1, 0, 0, 1]
        #     p.changeVisualShape(self.target, -1, rgbaColor=target_color)

        reward = self.task_success
        info = {
            'task_success': self.task_success,
            'target_index': self.target_index,

            'old_tool_pos': old_tool_pos.copy(),
            'tool_pos': self.tool_pos,
            'tool_orient': self.tool_orient,
            # 'rope_pos': rope_pos,

            'ground_truth': self.goal.copy(),
            'frechet': frechet
        }
        done = self.task_success

        return obs, reward, done, info

    def _get_obs(self, forces):
        robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices,
                                              physicsClientId=self.id)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])

        rope_pos = [
            p.getBasePositionAndOrientation(link)[0] for link in self.rope
        ]

        robot_obs = dict(
            raw_obs=np.concatenate([self.tool_pos, self.tool_orient, robot_joint_positions, *rope_pos]),
            hindsight_goal=np.concatenate([self.tool_pos, self.tool_pos]),
            goal=self.goal.copy(),
        )
        return robot_obs

    def reset(self):
        self.task_success = False

        """set up standard environment"""
        self.setup_timing()
        _human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, \
            _human_lower_limits, _human_upper_limits, self.robot_right_arm_joint_indices, \
            self.robot_left_arm_joint_indices, self.gender \
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

        """set up shelf environment objects"""
        self.table_pos = table_pos = np.array([0, -.9, 0])
        if self.stochastic:
            self.table_pos = table_pos = table_pos + self.table_noise

        self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'),
                                basePosition=table_pos, baseOrientation=default_orientation, physicsClientId=self.id)

        self.init_robot_arm()
        self.generate_target()

        wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, .1, 1])
        wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[4, .1, 1], rgbaColor=self.wall_color)

        wall_pos, wall_orient = np.array([0., -2., 1.]), np.array([0, 0, 0, 1])
        self.wall = p.createMultiBody(basePosition=wall_pos, baseOrientation=wall_orient,
                                      baseCollisionShapeIndex=wall_collision, baseVisualShapeIndex=wall_visual,
                                      physicsClientId=self.id)

        """configure pybullet"""
        p.setGravity(0, 0, -10, physicsClientId=self.id)
        p.setGravity(0, 0, 0, self.robot, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
        # Enable rendering
        p.resetDebugVisualizerCamera(cameraDistance=.1, cameraYaw=180, cameraPitch=-30,
                                     cameraTargetPosition=[0, -.25, 1.0], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.task_success = 0
        self.goal = np.concatenate(self.goal_points)
        self.curr_step = 0
        self.grab_constraints = []

        return self._get_obs([0])

    def init_start_pos(self):
        """exchange this function for curriculum"""
        self.init_pos = np.array([0, -.5, 1.1])

        if self.stochastic:
            self.init_pos += self.init_pos_random.uniform([-0.4, -0.1, -0.1], [0.4, 0.1, 0.1], size=3)

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
        if index is None:
            self.target_index = self.np_random.choice(self.target_indices)
        else:
            self.target_index = index

        self.unique_index = self.target_index

    def reset_noise(self):
        offset = self.np_random.choice((0.1, 0)) if self.table_offset is None else self.table_offset
        self.table_noise = self.np_random.uniform([-0.25, -0.05, 0], [0.15, 0.05, 0], size=3)
        self.table_noise[1] = self.table_noise[1] + offset

    def calibrate_mode(self, calibrate, split):
        self.wall_color = [255 / 255, 187 / 255, 120 / 255, 1] if calibrate else None
        self.table_offset = 0.1 if split else 0

    def create_shape(self, index):
        """create a shape that fits in a 1 x 1 box at z = 0"""
        # square
        # nodes = np.array([[.5, .5, 0], [.5, -.5, 0], [-.5, -.5, 0], [-.5, .5, 0]])
        # edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3]), (nodes[3], nodes[0])]
        # points_per_edge = num_points // len(edges)
        # for edge in edges:
        #     perimeter += norm(edge[0]-edge[1])
        #     for i in range(points_per_edge):
        #         point = (i / points_per_edge) * edge[0] + (1 - i / points_per_edge) * edge[1]
        #         goal_points.append(point)

        def ellipse():
            goal_points = []
            index = np.linspace(0, tau, num=self.num_links, endpoint=False)
            xrad = self.np_random.uniform(.25, 1)
            yrad = self.np_random.uniform(.25, 1)
            for i in index:
                goal_points.append(np.array([xrad*np.cos(i), yrad*np.sin(i), 0]))
            perimeter = np.pi*(3*(xrad+yrad) - np.sqrt((3*xrad+yrad)*(xrad+3*yrad)))  # ramanujan approximation
            return goal_points, perimeter

        def sinusoid():
            goal_points = []
            index = np.linspace(-np.pi, np.pi, num=self.num_links, endpoint=False)
            amp = self.np_random.uniform(.1, 10)
            rate = self.np_random.uniform(1, 2)
            for i in index:
                goal_points.append(np.array([i, amp*np.sin(rate*i), 0]))
            perimeter = 2*np.pi*np.sqrt(1+amp**2)
            return goal_points, perimeter

        goal_points, perimeter = [
            ellipse,
            sinusoid
        ][index]()
        return np.array(goal_points), perimeter

    def generate_target(self):
        table_center = (self.table_pos + np.array([0, .2, .7]))

        # Add beaded cable.
        length = 1
        num_parts = 100
        distance = length / num_parts
        radius = distance / 3
        part_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[radius, radius, radius])
        part_visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius * 10, rgbaColor=[1, .95, .9, 1])

        # Iterate and add to object_points, but interestingly, downstream code
        # only uses it for rewards, not for actions.
        self.rope = []
        # index = np.linspace(0, tau, num=num_parts, endpoint=False)
        index = np.linspace(-length/2, length/2, num=num_parts, endpoint=False)
        for i in index:
            # position = table_center + length / tau * np.array([np.cos(i), np.sin(i), 0])
            position = table_center + np.array([i, 0, 0])
            part_id = p.createMultiBody(0.01, part_shape, part_visual, basePosition=position)
            if len(self.rope) > 0:
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=self.rope[-1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=part_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=(0, 0, distance),
                    childFramePosition=(0, 0, 0))
                p.changeConstraint(constraint_id, maxForce=1000)
                p.changeDynamics(part_id, -1, lateralFriction=int(1e10), spinningFriction=int(1e10), rollingFriction=int(1e10), restitution=0)
            self.rope.append(part_id)

        self.goal_points, perimeter = self.create_shape(self.target_index)
        self.goal_points = self.goal_points * length / perimeter + table_center
        for goal_point in self.goal_points:
            sphere_collision = -1
            sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1],
                                                physicsClientId=self.id)
            target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                            baseVisualShapeIndex=sphere_visual, basePosition=goal_point,
                                            useMaximalCoordinates=False, physicsClientId=self.id)
        self.update_targets()

    def update_targets(self):
        pass

    @property
    def tool_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[0])

    @property
    def tool_orient(self):
        return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[1])
