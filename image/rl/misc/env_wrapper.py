from functools import reduce
import os
from pathlib import Path
import h5py

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
                'sim_keyboard': sim_keyboard,
                'dict_to_array': dict_to_array,
            }
            self.adapts = [adapt_map[adapt] for adapt in config['adapts']]
            self.adapts = [adapt(self, config) for adapt in self.adapts]
            self.adapt_step = lambda obs, r, done, info: reduce(lambda sub_tran, adapt: adapt._step(*sub_tran),
                                                                self.adapts, (obs, r, done, info))
            self.adapt_reset = lambda obs: reduce(lambda obs, adapt: adapt._reset(obs), 
                                                                self.adapts, obs)

        def step(self, action):
            tran = super().step(action)
            tran = self.adapt_step(*tran)
            return tran

        def reset(self):
            obs = super().reset()
            obs = self.adapt_reset(obs)
            return obs

    return Overhead(config)


class LibraryWrapper(Env):
    def __init__(self, config):
        self.env_name = config['env_name']
        self.base_env = {
            "OneSwitch": ag.OneSwitchJacoEnv,
            "Bottle": ag.BottleJacoEnv,
            "Valve": ag.ValveJacoEnv,
            "PointReach": ag.PointReachJacoEnv,
            "Rope": ag.RopeEnv,
            "BlockPush": ag.BlockPushEnv,
        }[config['env_name']]
        self.base_env = self.base_env(**config['env_kwargs'])
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self.feature_sizes = self.base_env.feature_sizes
        self.terminate_on_failure = config['terminate_on_failure']

    def step(self, action):
        obs, r, done, info = self.base_env.step(action)
        # done = info['task_success']
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
            self.action_grab = config.get('action_grab', False)
            self.action_space = {
                "trajectory": spaces.Box(-.1, .1, (3+self.action_grab,)),
                "joint": spaces.Box(-1, 1, (7+self.action_grab,)),
                "disc_traj": spaces.Box(0, 1, (6+self.action_grab,)),
            }[config['action_type']]
            self.translate = {
                # 'target': target,
                'trajectory': self.trajectory,
                'joint': self.joint,
                'disc_traj': self.disc_traj,
            }[config['action_type']]
            if self.action_grab:
                self.sub_translate = self.translate
                self.translate = self.grab

        def joint(self, action, info={}):
            clip_by_norm = lambda traj, limit: traj / max(1e-4, norm(traj)) * np.clip(norm(traj), None, limit)
            action = clip_by_norm(action, 1)
            info['joint'] = action
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

        def grab(self, action, info={}):
            info['grab'] = action[-1]
            joint_action, info = self.sub_translate(action[:-1], info)
            action = np.concatenate([joint_action, [action[-1]]])
            return action, info

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


class array_to_dict:
    def __init__(self, master_env, config):
        pass

    def _step(self, obs, r, done, info):
        if not isinstance(obs, dict):
            obs = {'raw_obs': obs}
        return obs, r, done, info

    def _reset(self, obs, info=None):
        if not isinstance(obs, dict):
            obs = {'raw_obs': obs}
        return obs


class oracle:
    def __init__(self, master_env, config):
        self.oracle_type = config['oracle']
        if 'model' in self.oracle_type:
            self.oracle = master_env.oracle = {
                "Bottle": BottleOracle,
            }[master_env.env_name](master_env.rng, **config['oracle_kwargs'])
        else:
            oracle_type = {
                'keyboard': KeyboardOracle,
                'dummy_gaze': BottleOracle,
            }[config['oracle']]
            if config['oracle'] == 'sim_gaze':
                self.oracle = master_env.oracle = oracle_type(**config['gaze_oracle_kwargs'])
            elif config['oracle'] == 'dummy_gaze':
                self.oracle = master_env.oracle = oracle_type(master_env.rng, **config['oracle_kwargs'])
            else:
                self.oracle = master_env.oracle = oracle_type()
        self.master_env = master_env
        del master_env.feature_sizes['goal']
        master_env.feature_sizes['recommend'] = self.oracle.size

    def _step(self, obs, r, done, info):
        self._predict(obs, info)
        return obs, r, done, info

    def _reset(self, obs, info=None):
        self.oracle.reset()
        self.master_env.recommend = obs['recommend'] = np.zeros(self.oracle.size)
        return obs

    def _predict(self, obs, info):
        recommend, _info = self.oracle.get_action(obs, info)
        self.master_env.recommend = obs['recommend'] = info['recommend'] = recommend
        info['noop'] = not self.oracle.status.curr_intervention


class goal:
    """
    Chooses what features from info to add to obs
    """

    def __init__(self, master_env, config):
        self.env_name = master_env.env_name
        self.master_env = master_env
        self.goal_feat_func = dict(
            # Kitchen=(lambda info: [info['target1_pos'], info['orders']],lambda info: [info['target1_pos'], info['orders'],info['tasks']]),
            Kitchen=lambda info: [info['target1_pos'], info['orders'], info['tasks']],
            # Kitchen=None,
            Bottle=None,
            OneSwitch=None,
            Valve=None,
            AnySwitch=lambda info: [info['switch_pos']],
            PointReach=lambda info: [info['sub_goal']],
            Rope=lambda info: [info['ground_truth']],
            BlockPush=lambda info: [info['ground_truth']]
        )[self.env_name]
        self.hindsight_feat = dict(
            # Kitchen=({'tool_pos': 3, 'orders': 2},{'tool_pos': 3, 'orders': 2, 'tasks': 6}),
            Kitchen={'tool_pos': 3, 'orders': 2, 'tasks': 6},
            Bottle={'tool_pos': 3},
            OneSwitch={'tool_pos': 3},
            Valve={'valve_angle': 2},
            AnySwitch={'tool_pos': 3},
            PointReach={'tool_pos': 3},
            Rope={'ground_truth': 300},
            BlockPush={'ground_truth': 3}
        )[self.env_name]
        # if isinstance(self.goal_feat_func,tuple):
        #     self.goal_feat_func = self.goal_feat_func[config['goal_func_ind']]
        #     self.hindsight_feat = self.hindsight_feat[config['goal_func_ind']]
        master_env.goal_size = self.goal_size = sum(self.hindsight_feat.values())

    def _step(self, obs, r, done, info):
        if self.goal_feat_func is not None:
            obs['goal'] = np.concatenate([np.ravel(state_component) for state_component in self.goal_feat_func(info)])

        hindsight_feat = np.concatenate(
            [np.ravel(info[state_component]) for state_component in self.hindsight_feat.keys()])

        obs['hindsight_goal'] = hindsight_feat
        return obs, r, done, info

    def _reset(self, obs, info=None):
        if self.goal_feat_func is not None:
            obs['goal'] = np.zeros(self.goal_size)

        obs['hindsight_goal'] = np.zeros(self.goal_size)
        return obs


class static_gaze:
    def __init__(self, master_env, config):
        self.gaze_dim = config['gaze_dim']
        del master_env.feature_sizes['goal']
        master_env.feature_sizes['gaze_features'] = self.gaze_dim
        self.env_name = master_env.env_name
        self.master_env = master_env
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

    def _reset(self, obs, info=None):
        if self.env_name == 'OneSwitch':
            index = self.master_env.base_env.target_indices.index(self.master_env.base_env.unique_index)
        elif self.env_name == 'Bottle':
            index = self.master_env.base_env.unique_index
        obs['gaze_features'] = self.static_gaze = self.sample_gaze(index)
        return obs


class real_gaze:
    def __init__(self, master_env, config):
        self.gaze_dim = config['gaze_dim']
        del master_env.feature_sizes['goal']
        master_env.feature_sizes['gaze_features'] = self.gaze_dim
        self.env_name = master_env.env_name
        self.master_env = master_env
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

    def _reset(self, obs, info=None):
        self.restart_gaze_thread()
        self.update_obs(obs)
        return obs


class sim_target:
    def __init__(self, master_env, config):
        self.env_name = master_env.env_name
        self.master_env = master_env
        self.feature = config.get('feature')
        del master_env.feature_sizes['goal']
        self.target_size = master_env.feature_sizes['target'] = 2 if self.env_name == 'Valve' else 3

        # should change to automate for all features eventually
        if self.feature == 'direction':
            self.target_size = master_env.feature_sizes['target'] = 3

        self.goal_noise_std = config['goal_noise_std']

    def _step(self, obs, r, done, info):
        self.add_target(obs, info)
        return obs, r, done, info

    def _reset(self, obs, info=None):
        self.add_target(obs, info)
        return obs

    def add_target(self, obs, info):
        if self.feature is None or self.feature is 'goal':
            target = obs['goal']
        elif info is None:
            target = np.zeros(self.target_size)
        else:
            target = info[self.feature]
        noise = np.random.normal(scale=self.goal_noise_std, size=target.shape) if self.goal_noise_std else 0
        obs['target'] = target + noise

class sim_keyboard:
    def __init__(self, master_env, config):
        self.env_name = master_env.env_name
        self.master_env = master_env
        self.feature = config.get('feature')
        del master_env.feature_sizes['goal']
        master_env.feature_sizes['target'] = 6

    def _step(self, obs, r, done, info):
        self.add_target(obs, info)
        return obs, r, done, info

    def _reset(self, obs, info=None):
        self.add_target(obs, info)
        return obs

    def add_target(self, obs, info):
        traj = obs[self.feature] - obs['tool_pos']
        axis = np.argmax(np.abs(traj))
        index = 2 * axis + (traj[axis] > 0)
        action = np.zeros(6)
        action[index] = 1
        obs['target'] = action

class joint:
    def __init__(self, master_env, config):
        master_env.observation_space = spaces.Box(-np.inf, np.inf, (master_env.observation_space.low.size + 7,))

    def _step(self, obs, r, done, info):
        obs['raw_obs'] = np.concatenate((obs['raw_obs'], obs['joint']))
        return obs, r, done, info

    def _reset(self, obs, info=None):
        obs['raw_obs'] = np.concatenate((obs['raw_obs'], obs['joint']))
        return obs


class dict_to_array:
    def __init__(self, master_env, config):
        pass
        # master_env.observation_space = spaces.Box(-np.inf,np.inf,(master_env.observation_space.low.size+3,))

    def _step(self, obs, r, done, info):
        obs = np.concatenate((obs['raw_obs'], obs['target']))
        return obs, r, done, info

    def _reset(self, obs, info=None):
        obs = np.concatenate((obs['raw_obs'], obs['target']))
        return obs


class reward:
    """ rewards capped at 'cap' """

    def __init__(self, master_env, config):
        self.range = (config['reward_min'], config['reward_max'])
        self.master_env = master_env
        self.reward_type = config.get('reward_type')
        self.reward_temp = config.get('reward_temp')
        self.reward_offset = config.get('reward_offset')

    def _step(self, obs, r, done, info):
        if self.reward_type == 'custom':
            r = -1
            r += np.exp(-norm(info['tool_pos'] - info['target1_pos'])) / 2
            if info['target1_reached']:
                r = -.5
                r += np.exp(-norm(info['tool_pos'] - info['target_pos'])) / 2
            if info['task_success']:
                r = 0
        elif self.reward_type == 'custom_kitchen':
            r = -1
            # print(
            # 	max(0,info['microwave_angle'] - -.7),
            # 	max(0,.7-info['fridge_angle']),
            # 	norm(info['tool_pos'] - info['target1_pos']),
            # 	norm(info['tool_pos'] - info['target_pos'])
            # 	)
            if not info['tasks'][0] and (info['orders'][0] == 0 or info['tasks'][1]):
                r += np.exp(-10 * max(0, info['microwave_angle'] - -.7)) / 6 * 3 / 4 * 1 / 2
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['microwave_handle'])) / 6 / 4 * 1 / 2
            elif info['tasks'][0]:
                r += 1 / 6
            if not info['tasks'][1] and (info['orders'][0] == 1 or info['tasks'][0]):
                r += np.exp(-10 * max(0, .7 - info['fridge_angle'])) / 6 * 3 / 4 * 1 / 2
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['fridge_handle'])) / 6 / 4 * 1 / 2
            elif info['tasks'][1]:
                r += 1 / 6

            if not info['tasks'][2] and info['tasks'][0] and info['tasks'][1]:
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['target1_pos'])) / 6 * 1 / 2
            elif info['tasks'][2]:
                r = -1 / 2
            if not info['tasks'][3] and info['tasks'][2]:
                r += np.exp(-self.reward_temp * norm(info['tool_pos'] - info['target_pos'])) / 6 * 1 / 2
            elif info['tasks'][3]:
                r = -1 / 3

            if not info['tasks'][4] and info['tasks'][3] and (info['orders'][1] == 0 or info['tasks'][5]):
                r += np.exp(-norm(info['microwave_angle'] - 0)) / 6 * 3 / 4 * 1 / 2
                dist = norm(info['tool_pos'] - info['microwave_handle'])
                if dist > .25:
                    r += np.exp(-self.reward_temp * dist) / 6 / 4 * 1 / 2
                else:
                    r += np.exp(-self.reward_temp * .25) / 6 / 4 * 1 / 2
            elif info['tasks'][4]:
                r += 1 / 6
            if not info['tasks'][5] and info['tasks'][3] and (info['orders'][1] == 1 or info['tasks'][4]):
                r += np.exp(-norm(info['fridge_angle'] - 0)) / 6 * 3 / 4 * 1 / 2
                dist = norm(info['tool_pos'] - info['fridge_handle'])
                if dist > .25:
                    r += np.exp(-self.reward_temp * dist) / 6 / 4 * 1 / 2
                else:
                    r += np.exp(-self.reward_temp * .25) / 6 / 4 * 1 / 2
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
            r = -1 + .5 * (info['task_success'] + info['target1_reached'])
        elif self.reward_type == 'terminal_interrupt':
            r = info['noop']
        # done = info['noop']
        elif self.reward_type == 'part_sparse_kitchen':
            r = -1 + sum(info['tasks']) / 6
        elif self.reward_type == 'valve_exp':
            dist = np.abs(self.master_env.base_env.angle_diff(info['valve_angle'], info['target_angle']))
            r = np.exp(-self.reward_temp * dist) - 1
        elif self.reward_type == 'pointreach_exp':
            r = -1
            r += np.exp(-norm(info['tool_pos'] - info['org_bottle_pos'])) / 2
            if info['target1_reached']:
                r = -.5
                r += np.exp(-norm(info['tool_pos'] - info['target_pos'])) / 2
            if info['task_success']:
                r = 0
        elif self.reward_type == 'rope_exp':
            r = -1
            r += np.exp(-info['frechet']) / 2
            if info['task_success']:
                r = 0
        elif self.reward_type == 'blockpush_exp':
            r = -1
            dist = norm(info['block_pos']-info['target_pos']) + norm(info['tool_pos'] - info['block_pos'])/2
            old_dist = norm(info['old_block_pos']-info['target_pos']) + norm(info['old_tool_pos'] - info['old_block_pos'])/2
            under_table_penalty = max(0, info['target_pos'][2]-info['tool_pos'][2]-.1)
            sigmoid = lambda x: 1/(1 + np.exp(-x))
            r += sigmoid(self.reward_temp*(old_dist-dist-under_table_penalty))*self.reward_offset
            if info['task_success']:
                r = 0
        else:
            raise Exception

        r = np.clip(r, *self.range)
        return obs, r, done, info

    def _reset(self, obs, info=None):
        return obs
