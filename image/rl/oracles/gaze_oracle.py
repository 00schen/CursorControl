import pybullet as p
import numpy as np
import torch
import cv2
from rl.gaze_capture.face_processor import FaceProcessor
from rl.gaze_capture.ITrackerModel import ITrackerModel
from .base_oracles import Oracle
from rlkit.util.io import load_local_or_remote_file
import threading
import random
import h5py


class KeyboardOracle(Oracle):
    def set_action(self):
        keys = p.getKeyboardEvents()
        inputs = {
            p.B3G_LEFT_ARROW: 'left',
            p.B3G_RIGHT_ARROW: 'right',
            ord('r'): 'forward',
            ord('f'): 'backward',
            p.B3G_UP_ARROW: 'up',
            p.B3G_DOWN_ARROW: 'down'
        }

        self.status.action = np.array([0, 0, 0, 0, 0, 0])

        for key in inputs:
            if key in keys and p.KEY_WAS_TRIGGERED:
                self.status.action = {
                    'left': np.array([0, 1, 0, 0, 0, 0]),
                    'right': np.array([1, 0, 0, 0, 0, 0]),
                    'forward': np.array([0, 0, 1, 0, 0, 0]),
                    'backward': np.array([0, 0, 0, 1, 0, 0]),
                    'up': np.array([0, 0, 0, 0, 0, 1]),
                    'down': np.array([0, 0, 0, 0, 1, 0]),
                    'noop': np.array([0, 0, 0, 0, 0, 0])
                }[inputs[key]]
                self.status.new_intervention = not self.status.curr_intervention
                self.status.curr_intervention = True
        if np.count_nonzero(self.status.action) == 0:
            self.status.new_intervention = False
            self.status.curr_intervention = False

    def reset(self):
        self.status.new_intervention = False
        self.status.curr_intervention = False


class SimGazeOracle(Oracle):
    def __init__(self, mode='il', gaze_demos_path=None, synth_gaze=False, per_step=False):
        super().__init__()
        self.from_gaze_demos = gaze_demos_path is not None
        self.synth_gaze = synth_gaze
        self.per_step = per_step

        if self.from_gaze_demos:
            self.data = []
            gaze_demos = load_local_or_remote_file(gaze_demos_path)
            for path in gaze_demos:
                self.data.append(path['env_infos'][0]['oracle_input'])
            if self.synth_gaze:
                data = np.array(self.data)
                self.mean = np.mean(data, axis=0)
                self.std = np.std(data, axis=0)

        else:
            data_path = {'il': 'image/rl/gaze_capture/gaze_data_il.h5',
                         'int': 'image/rl/gaze_capture/gaze_data_int.h5',
                         'rl': 'image/rl/gaze_capture/gaze_data_rl.h5'}[mode]

            self.data = h5py.File(data_path, 'r')

        self.size = 128
        self.gaze_input = None

    def get_gaze_input(self, info):
        if self.from_gaze_demos:
            if self.synth_gaze:
                self.gaze_input = np.maximum(np.random.normal(loc=self.mean, scale=self.std), 0)
            else:
                self.gaze_input = random.choice(self.data)
        else:
            target = np.where(info['target_string'] == 0)[0][0]
            self.gaze_input = random.choice(self.data[str(target)][()])

    def reset(self):
        self.status.new_intervention = False
        self.status.curr_intervention = False
        self.gaze_input = None


class RealGazeKeyboardOracle(KeyboardOracle):
    def __init__(self,
                 predictor_path='./image/rl/gaze_capture/model_files/shape_predictor_68_face_landmarks.dat'):
        super().__init__()
        self.size = 128
        self.webcam = cv2.VideoCapture(0)
        self.face_processor = FaceProcessor(predictor_path)

        self.i_tracker = ITrackerModel()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.i_tracker.cuda()
            state = torch.load('image/rl/gaze_capture/checkpoint.pth.tar')['state_dict']
        else:
            self.device = "cpu"
            state = torch.load('image/rl/gaze_capture/checkpoint.pth.tar',
                               map_location=torch.device('cpu'))['state_dict']
        self.i_tracker.load_state_dict(state, strict=False)
        self.input = np.zeros(self.size)
        self.gaze_thread = None

    def get_gaze_input(self):
        _, frame = self.webcam.read()
        features = self.face_processor.get_gaze_features(frame)

        if features is None:
            self.input = np.zeros(self.size)
        else:
            i_tracker_input = [torch.from_numpy(feature)[None].float().to(self.device) for feature in features]
            i_tracker_features = self.i_tracker(*i_tracker_input).detach().cpu().numpy()
            self.input = i_tracker_features[0]

    def get_action(self, obs, info=None):
        if self.gaze_thread is None or not self.gaze_thread.is_alive():
            self.gaze_thread = threading.Thread(target=self.get_gaze_input, name='gaze_thread')
            self.gaze_thread.start()
        self.set_action()
        return self.input, {}


class SimGazeModelOracle(SimGazeOracle):
    def __init__(self, base_oracle, mode='il', gaze_demos_path=None, per_step=False, thresh=1, inter_len=1, p=1):
        super().__init__(mode=mode, gaze_demos_path=gaze_demos_path, per_step=per_step)
        self.base_oracle = base_oracle
        # self.thresh = thresh
        # self.count = 0
        # self.inter = 0
        # self.inter_len = inter_len
        # self.p = p

    def get_action(self, obs, info=None):
        action, user_info = self.base_oracle.get_action(obs, info)

        # if self.inter > 0:
            # self.inter -= 1
            # self.status.new_intervention = False
            # self.status.curr_intervention = True

        # else:
        #     if np.count_nonzero(action) > 0:
        #         self.count += 1
        #         if self.count >= self.thresh:
        #             if np.random.rand() < self.p:
        #                 np.random.shuffle(action)
        #             self.status.action = action
        #             self.status.new_intervention = not self.status.curr_intervention
        #             self.status.curr_intervention = True
        #             self.inter = self.inter_len - 1
        #
        #     else:
        #         self.count = 0
        #         self.status.action = np.zeros(action.shape)
        #         self.status.new_intervention = False
        #         self.status.curr_intervention = False
        self.status.action = action
        if np.count_nonzero(action) > 0:
            self.status.new_intervention = not self.status.curr_intervention
            self.status.curr_intervention = True
        else:
            self.status.new_intervention = False
            self.status.curr_intervention = False

        if self.per_step or self.gaze_input is None:
            self.get_gaze_input(info)

        return self.gaze_input, user_info

    def reset(self):
        super().reset()
        self.inter = 0
        self.count = 0
        self.base_oracle.reset()


class SimGazeKeyboardOracle(KeyboardOracle, SimGazeOracle):
    def get_action(self, obs, info=None):
        self.set_action()
        if self.per_step or self.gaze_input is None:
            self.get_gaze_input(info)
        return self.gaze_input, {}

    def reset(self):
        SimGazeOracle.reset(self)


class RealGazeModelOracle(RealGazeKeyboardOracle):
    def __init__(self, base_oracle, predictor_path='./image/rl/gaze_capture/model_files/shape_predictor_68_face_landmarks.dat'):
        super().__init__(predictor_path)
        self.base_oracle = base_oracle
        self.size = 128

    def get_action(self, obs, info=None):
        if self.gaze_thread is None or not self.gaze_thread.is_alive():
            self.gaze_thread = threading.Thread(target=self.get_gaze_input, name='gaze_thread')
            self.gaze_thread.start()

        action, user_info = self.base_oracle.get_action(obs, info)
        self.status.action = action

        # self.status.new_intervention = np.count_nonzero(action) > 0

        if np.count_nonzero(action) > 0:
            self.status.new_intervention = not self.status.curr_intervention
            self.status.curr_intervention = True
        else:
            self.status.new_intervention = False
            self.status.curr_intervention = False

        return self.input, user_info

    def reset(self):
        super().reset()
        self.base_oracle.reset()
