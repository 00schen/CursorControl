from .base_oracles import UserModelOracle
import numpy as np
from numpy.linalg import norm


class StraightLineOracle(UserModelOracle):
    def _query(self, obs, info):
        target_pos = info['target_pos']
        old_traj = target_pos - info['old_tool_pos']
        new_traj = info['tool_pos'] - info['old_tool_pos']
        info['cos_error'] = np.dot(old_traj, new_traj) / (norm(old_traj) * norm(new_traj))
        criterion = info['cos_error'] < self.threshold

        return criterion, target_pos
