import numpy as np


class IdentityPolicy:
    """Assuming simulated keyboard, directly return the target value"""
    def __init__(self):
        pass

    def get_action(self, obs):
        action = obs['target']
        return action, {}

    def reset(self):
        pass
