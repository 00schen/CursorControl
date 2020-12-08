import numpy as np
import pybullet as p


class DefaultGazePolicy:
    def __init__(self):
        self.action = np.zeros(6)

    def get_action(self, obs):
        keys = p.getKeyboardEvents()
        inputs = {
            p.B3G_LEFT_ARROW: 'left',
            p.B3G_RIGHT_ARROW: 'right',
            ord('r'): 'forward',
            ord('f'): 'backward',
            p.B3G_UP_ARROW: 'up',
            p.B3G_DOWN_ARROW: 'down'
        }

        for key in inputs:
            if key in keys and p.KEY_WAS_TRIGGERED:
                self.action = {
                    'left': np.array([0, 1, 0, 0, 0, 0]),
                    'right': np.array([1, 0, 0, 0, 0, 0]),
                    'forward': np.array([0, 0, 1, 0, 0, 0]),
                    'backward': np.array([0, 0, 0, 1, 0, 0]),
                    'up': np.array([0, 0, 0, 0, 0, 1]),
                    'down': np.array([0, 0, 0, 0, 1, 0]),
                    'noop': np.array([0, 0, 0, 0, 0, 0])
                }[inputs[key]]
        print(self.action)
        return self.action, {}

    def reset(self):
        self.action = np.zeros(6)
