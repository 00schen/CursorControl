import numpy as np


class UserInputPolicy:
    def __init__(self, env, p, base_policy=None, intervene=True):
        self.oracle_status = env.oracle.status
        self.action = np.zeros(6)
        self.rng = env.rng
        self.p = p
        self.base_policy = base_policy
        self.intervene = intervene

    def get_action(self, obs):
        if self.base_policy is not None:
            self.action, info = self.base_policy.get_action(obs)
        if self.intervene and np.count_nonzero(self.oracle_status.action) > 0:
            self.action = self.oracle_status.action
            if self.rng.random() > self.p:
                self.action = np.zeros(6)
                self.action[self.rng.choice(6)] = 1
        return self.action, {}

    def reset(self):
        self.action = np.zeros(6)
