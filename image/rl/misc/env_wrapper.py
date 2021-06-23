from functools import reduce
import os
from pathlib import Path
import h5py
from collections import OrderedDict

import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm

import pybullet as p
import assistive_gym as ag
from gym import spaces, Env

import cv2
import torch
from gaze_capture.face_processor import FaceProcessor
from gaze_capture.ITrackerModel import ITrackerModel
import threading
from rl.oracles import *

main_dir = str(Path(__file__).resolve().parents[2])


def default_overhead(config):
    factory_map = {
        'session': session_factory,
    }
    factories = [factory_map[factory] for factory in config['factories']]
    factories = [action_factory] + factories
    wrapper = reduce(lambda value, func: func(value), factories, LibraryWrapper)

    class Overhead(wrapper):
        def __init__(self, config):
            super().__init__(config)
            self.rng = default_rng(config['seedid'])
            adapt_map = {
                'static_gaze': static_gaze,
                'real_gaze': real_gaze,
                'joint': joint,
                'goal': goal,
                'reward': reward,
                'sim_target': sim_target,
                'dict_to_array': dict_to_array,
            }
            self.adapts = [adapt_map[adapt] for adapt in config['adapts']]
            self.adapts = [adapt(self, config) for adapt in self.adapts]
            self.adapt_step = lambda obs, r, done, info: reduce(lambda sub_tran, adapt: adapt._step(*sub_tran),
                                                                self.adapts, (obs, r, done, info))
            self.adapt_reset = lambda obs: reduce(lambda obs, adapt: adapt._reset(obs), 
                                                                self.adapts, obs)
            # Action space set by Action class
            # Observation space size set by LibraryWrapper class
            self.observation_space = spaces.Box(-np.inf, np.inf, (sum(self.feature_sizes.values()),))
            # Goal Space size set by adapt classes
            self.goal_space = spaces.Box(-np.inf, np.inf, (sum(self.goal_feat_sizes.values()),))

        def step(self, action):
            tran = super().step(action)
            obs, r, done, info = self.adapt_step(*tran)
            obs['base_obs'] = np.concatenate([np.array([obs[feat]]).ravel() for feat in self.feature_sizes.keys()])
            obs['goal_obs'] = np.concatenate([np.array([obs[feat]]).ravel() for feat in self.goal_feat_sizes.keys()])
            return tran

        def reset(self):
            obs = super().reset()
            obs = self.adapt_reset(obs)
            obs['base_obs'] = np.concatenate([np.array([obs[feat]]).ravel() for feat in self.feature_sizes.keys()])
            obs['goal_obs'] = np.concatenate([np.array([obs[feat]]).ravel() for feat in self.goal_feat_sizes.keys()])
            return obs

    return Overhead(config)


class LibraryWrapper(Env):
    def __init__(self, config):
        self.env_name = config['env_name']
        self.base_env = {
            "Laptop": ag.LaptopJacoEnv,
            "OneSwitch": ag.OneSwitchJacoEnv,
            # "ThreeSwitch": ag.ThreeSwitchJacoEnv,
            "AnySwitch": ag.AnySwitchJacoEnv,
            "Bottle": ag.BottleJacoEnv,
            "Kitchen": ag.KitchenJacoEnv,
        }[config['env_name']]
        self.base_env = self.base_env(**config['env_kwargs'])
        self.terminate_on_failure = config['terminate_on_failure']

        self.feature_sizes = {
            "Laptop": OrderedDict(), # env not used
            "OneSwitch": OrderedDict({'tool_pos':3, 'tool_orient': 4, 'goal_set': 20}),
            "ThreeSwitch": OrderedDict(), # env not used
            "AnySwitch": OrderedDict(), # env not used
            "Bottle": OrderedDict({'tool_pos':3, 'tool_orient': 4, 'shelf_pos': 3}),
            "Kitchen": OrderedDict({'tool_pos':3, 'tool_orient': 4,
                                    'microwave_handle': 3, 'fridge_handle': 3,
                                    'microwave_angle': 1, 'microwave_angle': 1}),
        }[config['env_name']]

        self.base_goal_size = sum(self.base_env.goal_feat_sizes.values())

    def step(self, action):
        obs, r, done, info = self.base_env.step(action)
        done = info['task_success']
        if self.terminate_on_failure and hasattr(self.base_env, 'wrong_goal_reached'):
            done = done or self.base_env.wrong_goal_reached()
        return obs, r, done, info

    def reset(self):
        return self.base_env.reset()

    def render(self, mode=None, **kwargs):
        return self.base_env.render(mode)

    def seed(self, value):
        self.base_env.seed(value)

    def close(self):
        self.base_env.close()

    def get_base_env(self):
        return self.base_env


def action_factory(base):
    class Action(base):
        def __init__(self, config):
            super().__init__(config)
            self.action_type = config['action_type']
            self.action_space = {
                "trajectory": spaces.Box(-.1, .1, (3,)),
                "joint": spaces.Box(-.25, .25, (7,)),
                "disc_traj": spaces.Box(0, 1, (6,)),
            }[config['action_type']]
            self.translate = {
                # 'target': target,
                'trajectory': self.trajectory,
                'joint': self.joint,
                'disc_traj': self.disc_traj,
            }[config['action_type']]
            self.smooth_alpha = config['smooth_alpha']

        def joint(self, action, info={}):
            clip_by_norm = lambda traj, limit: traj / max(1e-4, norm(traj)) * np.clip(norm(traj), None, limit)
            action = clip_by_norm(action, .25)
            self.action = self.smooth_alpha * action + (1 - self.smooth_alpha) * self.action if np.count_nonzero(
                self.action) else action
            info['joint'] = self.action
            return action, info

        def target(self, coor, info={}):
            base_env = self.base_env
            info['target'] = coor
            joint_states = p.getJointStates(base_env.robot, jointIndices=base_env.robot_left_arm_joint_indices,
                                            physicsClientId=base_env.id)
            joint_positions = np.array([x[0] for x in joint_states])

            link_pos = p.getLinkState(base_env.robot, 13, computeForwardKinematics=True, physicsClientId=base_env.id)[0]
            new_pos = np.array(coor) + np.array(link_pos) - base_env.tool_pos

            new_joint_positions = np.array(
                p.calculateInverseKinematics(base_env.robot, 13, new_pos, physicsClientId=base_env.id))
            new_joint_positions = new_joint_positions[:7]
            action = new_joint_positions - joint_positions
            return self.joint(action, info)

        def trajectory(self, traj, info={}):
            clip_by_norm = lambda traj, min_l=None, max_l=None: traj / max(1e-4, norm(traj)) * np.clip(norm(traj),
                                                                                                       min_l, max_l)
            traj = clip_by_norm(traj, .07, .1)
            info['trajectory'] = traj
            return self.target(self.base_env.tool_pos + traj, info)

        def disc_traj(self, onehot, info={}):
            info['disc_traj'] = onehot
            index = np.argmax(onehot)
            traj = [
                np.array((-1, 0, 0)),
                np.array((1, 0, 0)),
                np.array((0, -1, 0)),
                np.array((0, 1, 0)),
                np.array((0, 0, -1)),
                np.array((0, 0, 1)),
            ][index]
            return self.trajectory(traj, info)

        def step(self, action):
            action, ainfo = self.translate(action)
            obs, r, done, info = super().step(action)
            info = {**info, **ainfo}
            return obs, r, done, info

        def reset(self):
            self.action = np.zeros(7)
            return super().reset()

    return Action


def session_factory(base):
    class Session(base):
        def __init__(self, config):
            config['env_kwargs']['session_goal'] = True
            super().__init__(config)
            self.goal_reached = False

        def new_goal(self, index=None):
            self.base_env.set_target_index(index)
            self.base_env.reset_noise()
            self.goal_reached = False

        def step(self, action):
            o, r, d, info = super().step(action)
            if info['task_success']:
                self.goal_reached = True
            return o, r, d, info

        def reset(self):
            return super().reset()

    return Session


class Adapter:
    def __init__(self, master_env, config):
        self.env_name = master_env.env_name
        self.master_env = master_env

    def _step(self, obs, r, done, info):
        return obs, r, done, info

    def _reset(self, obs):
        return obs
        

class goal(Adapter):
    """
    Chooses what features from info to add to obs
    """

    def __init__(self, master_env, config):
        super().__init__(master_env,config)

        master_env.goal_feat_sizes = OrderedDict({'goal': sum(master_env.base_env.goal_feat_sizes.values())})


class static_gaze(Adapter):
    def __init__(self, master_env, config):
        super().__init__(master_env,config)

        self.gaze_dim = config['gaze_dim']
        master_env.goal_feat_sizes = OrderedDict({'gaze_features': self.gaze_dim})
        with h5py.File(os.path.join(str(Path(__file__).resolve().parents[2]), 'gaze_capture', 'gaze_data',
                                    config['gaze_path']), 'r') as gaze_data:
            self.gaze_dataset = {k: v[()] for k, v in gaze_data.items()}
        self.per_step = True

    def sample_gaze(self, index):
        unique_target_index = index
        data = self.gaze_dataset[str(unique_target_index)]
        return self.master_env.rng.choice(data)

    def _step(self, obs, r, done, info):
        if self.per_step:
            if self.env_name == 'OneSwitch':
                self.static_gaze = self.sample_gaze(self.master_env.base_env.target_indices.index(info['unique_index']))
            elif self.env_name == 'Bottle':
                self.static_gaze = self.sample_gaze(info['unique_index'])
        obs['gaze_features'] = self.static_gaze
        return obs, r, done, info

    def _reset(self, obs):
        if self.env_name == 'OneSwitch':
            index = self.master_env.base_env.target_indices.index(self.master_env.base_env.unique_index)
        elif self.env_name == 'Bottle':
            index = self.master_env.base_env.unique_index
        obs['gaze_features'] = self.static_gaze = self.sample_gaze(index)
        return obs


class real_gaze(Adapter):
    def __init__(self, master_env, config):
        super().__init__(master_env,config)
        self.gaze_dim = config['gaze_dim']

        master_env.goal_feat_sizes = OrderedDict({'gaze_features': self.gaze_dim})
        self.webcam = cv2.VideoCapture(0)
        self.face_processor = FaceProcessor(
            os.path.join(main_dir, 'gaze_capture', 'model_files', 'shape_predictor_68_face_landmarks.dat'))

        self.i_tracker = ITrackerModel()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.i_tracker.cuda()
            state = torch.load(os.path.join(main_dir, 'gaze_capture', 'checkpoint.pth.tar'))['state_dict']
        else:
            self.device = "cpu"
            state = torch.load(os.path.join(main_dir, 'gaze_capture', 'checkpoint.pth.tar'),
                               map_location=torch.device('cpu'))['state_dict']
        self.i_tracker.load_state_dict(state, strict=False)

        self.gaze = np.zeros(self.gaze_dim)
        self.gaze_lock = threading.Lock()
        self.gaze_thread = None

    def record_gaze(self):
        _, frame = self.webcam.read()
        features = self.face_processor.get_gaze_features(frame)

        if features is None:
            gaze = np.zeros(self.gaze_dim)
        else:
            i_tracker_input = [torch.from_numpy(feature)[None].float().to(self.device) for feature in features]
            i_tracker_features = self.i_tracker(*i_tracker_input).detach().cpu().numpy()
            gaze = i_tracker_features[0]

        self.gaze_lock.acquire()
        self.gaze = gaze
        self.gaze_lock.release()

    def restart_gaze_thread(self):
        if self.gaze_thread is None or not self.gaze_thread.is_alive():
            self.gaze_thread = threading.Thread(target=self.record_gaze, name='gaze_thread')
            self.gaze_thread.start()

    def update_obs(self, obs):
        self.gaze_lock.acquire()
        obs['gaze_features'] = self.gaze
        self.gaze_lock.release()

    def _step(self, obs, r, done, info):
        self.restart_gaze_thread()
        self.update_obs(obs)
        return obs, r, done, info

    def _reset(self, obs):
        self.restart_gaze_thread()
        self.update_obs(obs)
        return obs


class sim_target(Adapter):
    def __init__(self, master_env, config):
        super().__init__(master_env,config)

        self.feature = config.get('feature')
        self.target_size = 3  # ok for bottle and light switch, may not be true for other envs
        master_env.goal_feat_sizes = OrderedDict({'gaze_features': self.target_size})
        self.goal_noise_std = config['goal_noise_std']

    def _step(self, obs, r, done, info):
        self.add_target(obs)
        return obs, r, done, info

    def _reset(self, obs):
        self.add_target(obs)
        return obs

    def add_target(self, obs):
        if self.feature is None or self.feature is 'goal':
            target = obs['goal']
        else:
            target = obs[self.feature]
        noise = np.random.normal(scale=self.goal_noise_std, size=target.shape) if self.goal_noise_std else 0
        obs['gaze_features'] = target + noise

class joint(Adapter):
    def __init__(self, master_env, config):
        super().__init__(master_env,config)
        master_env.feature_sizes['joints'] = 7


class goal_set(Adapter):
    def __init__(self, master_env, config):
        super().__init__(master_env,config)
        master_env.feature_sizes['goal_set'] = 7


class dict_to_array(Adapter):
    def __init__(self, master_env, config):
        super().__init__(master_env,config)

    def _step(self, obs, r, done, info):
        obs = np.concatenate((obs['base_obs'],obs['target']))
        return obs, r, done, info
    
    def _reset(self, obs):
        obs = np.concatenate((obs['base_obs'],obs['target']))
        return obs


class reward(Adapter):
    """ rewards capped at 'cap' """

    def __init__(self, master_env, config):
        super().__init__(master_env,config)

        self.reward_type = config['reward_type']
        self.reward_temp = config['reward_temp']
        self.reward_offset = config['reward_offset']

    def _step(self, obs, r, done, info):
        if self.reward_type == 'custom':
            r = -1
            r += np.exp(-norm(info['tool_pos'] - info['target1_pos'])) / 2
            if info['target1_reached']:
                r = -.5
                r += np.exp(-norm(info['tool_pos'] - info['target_pos'])) / 2
            if info['task_success']:
                r = 0
        elif self.reward_type == 'kitchen_debug':
            r = -1
            # r += max(0, info['microwave_angle'] - -1.3)/1.3 * 3 / 4 * 1/2
            r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['microwave_handle'])) * 1/2
            if info['task_success']:
                r = 0
        elif self.reward_type == 'custom_kitchen':
            r = -1
            if not info['tasks'][0] and (info['orders'][0] == 0 or info['tasks'][1]):
                r += np.exp(-10 * max(0, info['microwave_angle'] - -.7)) / 6 * 3 / 4 * 1/2
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['microwave_handle'])) / 6 / 4 * 1/2
            elif info['tasks'][0]:
                r += 1 / 6
            if not info['tasks'][1] and (info['orders'][0] == 1 or info['tasks'][0]):
                r += np.exp(-10 * max(0, .7 - info['fridge_angle'])) / 6 * 3 / 4 * 1/2
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['fridge_handle'])) / 6 / 4 * 1/2
            elif info['tasks'][1]:
                r += 1 / 6

            if not info['tasks'][2] and info['tasks'][0] and info['tasks'][1]:
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['target1_pos'])) / 6 * 1/2
            elif info['tasks'][2]:
                r = -1 / 2
            if not info['tasks'][3] and info['tasks'][2]:
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['target_pos'])) / 6 * 1/2
            elif info['tasks'][3]:
                r = -1 / 3

            if not info['tasks'][4] and info['tasks'][3] and (info['orders'][1] == 0 or info['tasks'][5]):
                r += np.exp(-norm(info['microwave_angle'] - 0)) / 6 * 3 / 4 * 1/2
                dist = norm(info['tool_pos'] - info['microwave_handle'])
                if dist > .25:
                    r += np.exp(-self.reward_temp * dist) / 6 / 4 * 1/2
                else:
                    r += np.exp(-self.reward_temp * .25) / 6 / 4 * 1/2
            elif info['tasks'][4]:
                r += 1 / 6
            if not info['tasks'][5] and info['tasks'][3] and (info['orders'][1] == 1 or info['tasks'][4]):
                r += np.exp(-norm(info['fridge_angle'] - 0)) / 6 * 3 / 4 * 1/2
                dist = norm(info['tool_pos'] - info['fridge_handle'])
                if dist > .25:
                    r += np.exp(-self.reward_temp * dist) / 6 / 4 * 1/2
                else:
                    r += np.exp(-self.reward_temp * .25) / 6 / 4 * 1/2
            elif info['tasks'][5]:
                r += 1 / 6

            if info['task_success']:
                r = 0

        elif self.reward_type == 'dist':
            r = 0
            if not info['task_success']:
                dist = np.linalg.norm(info['tool_pos'] - info['target_pos'])
                r = np.exp(-self.reward_temp * dist + np.log(1 + self.reward_offset)) - 1
        elif self.reward_type == 'custom_switch':
            r = 0
            if not info['task_success']:
                dist = np.linalg.norm(info['tool_pos'] - info['switch_pos'][info['target_index']])
                r = np.exp(-self.reward_temp * dist + np.log(1 + self.reward_offset)) - 1

        elif self.reward_type == 'sparse':
            r = -1 + info['task_success']
        elif self.reward_type == 'part_sparse':
            r = -1 + .5 * (info['task_success'] + info['door_open'])
        elif self.reward_type == 'part_sparse_kitchen':
            r = -1 + sum(info['tasks']) / 6
        else:
            raise Exception

        return obs, r, done, info

