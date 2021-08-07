import pybullet as p
import numpy as np
from collections import deque
from numpy.linalg import norm
from .base_oracles import Oracle

class PointReachOracle(Oracle):
    # def get_action(self, obs):
    #     tool_pos = obs['tool_pos']

    #     if not obs['target1_reached']:
    #         target_pos = obs['org_bottle_pos']
    #     else:
    #         target_pos = obs['target_pos']

    #     traj = target_pos - tool_pos
    #     return traj, {}

    def get_action(self, obs):
        tool_pos = obs['raw_obs'][:3]

        if not obs['raw_obs'][7]:
            target_pos = obs['goal'][3:]
        else:
            target_pos = obs['goal'][:3]

        traj = target_pos - tool_pos
        

        return traj, {}