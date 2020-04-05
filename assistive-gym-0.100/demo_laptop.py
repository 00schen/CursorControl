from assistive_gym import LaptopEnv
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from stable_baselines import SAC

import numpy as np

model_path = "trained_models/ppo/FeedingJaco-v0.pt"
predictor_path = "../logs/sac_penalty_100/rl_model_500000_steps"

if __name__ == '__main__':
    env = LaptopEnv()
    env.reset()
    while True:
      1+1