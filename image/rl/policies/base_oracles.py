import numpy as np

class Oracle:
    def __init__(self):
        self.size = 6
        self.threshold = .6

    def _query(self,obs):
        pass

    def get_action(self, obs):
        criterion, target_pos = self._query(obs)
        action = np.zeros(self.size)
        if np.random.random() < criterion:
            traj = target_pos - obs['tool_pos']
            axis = np.argmax(np.abs(traj))
            index = 2 * axis + (traj[axis] > 0)
            action[index] = 1
        return action, {}

    def reset(self):
        pass
