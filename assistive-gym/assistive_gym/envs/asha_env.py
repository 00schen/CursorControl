import os
from gym import spaces
import numpy as np
import pybullet as p
from numpy.linalg import norm
from .env import AssistiveEnv
from gym.utils import seeding

class AshaEnv(AssistiveEnv):
    def __init__(self, success_dist=.03, session_goal=False, target_indices=None,
                    stochastic=True, robot_type='jaco', debug=False,
                    capture_frames=False, frame_skip=5):
        super().__init__(robot_type=robot_type, task='reaching', frame_skip=frame_skip,
                                             time_step=0.02, action_robot_len=7, obs_robot_len=7)
        self.success_dist = success_dist
        self.capture_frames = capture_frames
        self.debug = debug
        self.stochastic = stochastic
        self.session_goal = session_goal

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.init_pos_random, _ = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.curr_step += 1

        self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))
        self._step()

        obs = self._get_obs()
        info = obs
        if self.capture_frames:
            frame = self.get_frame()
            info['frame'] = frame
        reward = 0
        done = False
        return obs, reward, done, info

    def _get_obs(self):
        robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices,
                                              physicsClientId=self.id)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])
        _, switch_orient = p.getBasePositionAndOrientation(self.wall, physicsClientId=self.id)

        obs = {
            'task_success': self.task_success,
            'tool_pos': self.tool_pos,
            'tool_orient': self.tool_orient,

            'target_index': self.target_index,
            'goal':self.goal.copy(),
            'joints': robot_joint_positions,
        }
        return obs

    def reset(self):
        self.task_success = False

        """set up standard environment"""
        self.setup_timing()
        _human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, _human_lower_limits, _human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender \
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
        self.init_robot_arm()
        self.human_controllable_joint_indices = []
        self.human_lower_limits = np.array([])
        self.human_upper_limits = np.array([])
        
        """set random variables"""
        if not self.session_goal:
            self.set_target_index()  # instance override in demos
            self.reset_noise()

        wall_pos, wall_orient = np.array([0., -1., 1.]), [0, 0, 0, 1]
        if self.stochastic:
            wall_pos = wall_pos + self.wall_noise
        wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, .1, 1])
        wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, .1, 1], rgbaColor=self.wall_color)
        self.wall = p.createMultiBody(basePosition=wall_pos, baseOrientation=wall_orient,
                                      baseCollisionShapeIndex=wall_collision, baseVisualShapeIndex=wall_visual,
                                      physicsClientId=self.id)

        self._reset()

        """configure pybullet"""
        p.setGravity(0, 0, 0, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
        # Enable rendering
        p.resetDebugVisualizerCamera(physicsClientId=self.id, **self.camera_setting)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return self._get_obs()
    
    def init_robot_arm(self):
        self.init_start_pos()
        init_orient = p.getQuaternionFromEuler(np.array([0, np.pi / 2.0, 0]), physicsClientId=self.id)
        self.util.ik_random_restarts(self.robot, 11, self.init_pos, init_orient, self.world_creation,
                                     self.robot_left_arm_joint_indices, self.robot_lower_limits,
                                     self.robot_upper_limits,
                                     ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=100, max_ik_random_restarts=10,
                                     random_restart_threshold=0.03, step_sim=True)

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

    def calibrate_mode(self, calibrate, split):
        self.scene_offset = 0.1 if split else 0
        self.wall_color = [255 / 255, 187 / 255, 120 / 255, 1] if calibrate else None

    @property
    def tool_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[0])

    @property
    def tool_orient(self):
        return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[1])
