import numpy as np
import pybullet as p


class DefaultGazePolicy:
    def __init__(self, env, oracle_status, p):
        self.oracle_status = oracle_status
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


class OverrideGazePolicy:
    def __init__(self, policy, oracle_status):
        self.policy = policy
        self.oracle_status = oracle_status
        self.action = np.zeros(6)

    def get_action(self, obs):
        if np.count_nonzero(self.oracle_status.action) > 0:
            return self.oracle_status.action, {}

        return self.policy.get_action(obs)

    def reset(self):
        self.policy.reset()
        self.action = np.zeros(6)
        self.oracle_status.curr_intervention = False
        self.oracle_status.new_intervention = False
