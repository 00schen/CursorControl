import pybullet as p
import numpy as np
import torch
import cv2
from gaze_capture.face_processor import FaceProcessor
from gaze_capture.ITrackerModel import ITrackerModel
from .base_oracles import Oracle


class GazeOracle(Oracle):
    def __init__(self, env, predictor_path='./gaze_capture/data/shape_predictor_68_face_landmarks.dat'):
        super().__init__()
        self.size = 128
        self.env = env
        self.webcam = cv2.VideoCapture(0)
        self.face_processor = FaceProcessor(predictor_path)

        self.i_tracker = ITrackerModel()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.i_tracker.cuda()
            state = torch.load('gaze_capture/checkpoint.pth.tar')['state_dict']
        else:
            self.device = "cpu"
            state = torch.load('gaze_capture/checkpoint.pth.tar', map_location=torch.device('cpu'))['state_dict']
        self.i_tracker.load_state_dict(state, strict=False)

    def get_action(self, obs, info=None):
        keys = p.getKeyboardEvents()
        if p.B3G_SPACE in keys and p.KEY_WAS_TRIGGERED:
            _, frame = self.webcam.read()
            features = self.face_processor.get_gaze_features(frame)

            if features is None:
                input = np.zeros(self.size)
            else:
                i_tracker_input = [torch.from_numpy(feature)[None].float().to(self.device) for feature in features]
                i_tracker_features = self.i_tracker(*i_tracker_input).detach().cpu().numpy()
                input = i_tracker_features[0]

        else:
            input = np.zeros(self.size)

        return input, {}
