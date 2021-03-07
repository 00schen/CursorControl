import numpy as np


class UserInputPolicy:
    def __init__(self, env, p):
        self.oracle_status = env.oracle.status
        self.action = np.zeros(6)
        self.rng = env.rng
        self.p = p

    def get_action(self, obs):
        if self.rng.random() > self.p:
            self.action = np.zeros(6)
            self.action[self.rng.choice(6)] = 1
        if np.count_nonzero(self.oracle_status.action) > 0:
            self.action = self.oracle_status.action
        return self.action, {}

    def reset(self):
        self.action = np.zeros(6)
        self.oracle_status.curr_intervention = False
        self.oracle_status.new_intervention = False
