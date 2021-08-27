import pybullet as p
import numpy as np
from collections import deque
from numpy.linalg import norm
from .base_oracles import Oracle

class BlockPushPolicy(Oracle):
    # def get_action(self, obs):
    #     tool_pos = obs['tool_pos']

    #     if not obs['target1_reached']:
    #         target_pos = obs['org_bottle_pos']
    #     else:
    #         target_pos = obs['target_pos']

    #     traj = target_pos - tool_pos
    #     return traj, {}

    def get_action(self, obs):
        traj = obs['goal'] - obs['block_pos']
        force = traj*50
        action = np.concatenate((force[:2],[10], np.zeros(4)))

        return action, {}