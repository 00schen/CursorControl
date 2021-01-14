import pybullet as p
import numpy as np
import torch
import cv2
from rl.gaze_capture.face_processor import FaceProcessor
from rl.gaze_capture.ITrackerModel import ITrackerModel
from .base_oracles import Oracle, OracleStatus
from .three_switch_oracle import ThreeSwitchOracle
import threading
import random
import h5py


class GazeOracle(Oracle):
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
        self.status = OracleStatus()

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

        return self.input, {}


class SimGazeOracle(Oracle):
    def __init__(self, base_oracle, data_path='image/rl/gaze_capture/gaze_data.h5'):
        super().__init__()
        self.data = h5py.File(data_path, 'r')
        self.status = OracleStatus()
        self.size = 3
        self.base_oracle = base_oracle

    def get_action(self, obs, info=None):
        action, user_info = self.base_oracle.get_action(obs, info)
        self.status.action = action
        self.status.new_intervention = np.count_nonzero(action) > 0

        # if np.count_nonzero(action) > 0:
        #     self.status.new_intervention = not self.status.curr_intervention
        #     self.status.curr_intervention = True
        # else:
        #     self.status.new_intervention = False
        #     self.status.curr_intervention = False

        target_indices = np.nonzero(np.not_equal(info['target_string'], info['current_string']))[0]
        if len(target_indices) > 0:
            target_index = target_indices[0]
        else:
            target_index = np.random.choice(len(self.data.keys()))
        gaze_input = random.choice(self.data[str(target_index)][()])
        return gaze_input, user_info

    def reset(self):
        self.base_oracle.reset()


class GazeModelOracle(GazeOracle):
    def __init__(self, base_oracle, predictor_path='./image/rl/gaze_capture/model_files/shape_predictor_68_face_landmarks.dat'):
        super().__init__(predictor_path)
        self.base_oracle = base_oracle

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
        self.base_oracle.reset()
