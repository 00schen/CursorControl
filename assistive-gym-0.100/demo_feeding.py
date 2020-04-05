from assistive_gym import FeedingJacoOracleEnv,FeedingPretrainAgent,FeedingTwoDAgent
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from stable_baselines import SAC

import numpy as np

model_path = "trained_models/ppo/FeedingJaco-v0.pt"
predictor_path = "../logs/sac_penalty_100/rl_model_500000_steps"

if __name__ == '__main__':
  for _i in range(100):
    oracle = FeedingPretrainAgent(torch.load(model_path)[0])
    env = FeedingJacoOracleEnv()
    env.render()
    obs = env.reset()
    agent = FeedingTwoDAgent(env,oracle,SAC.load(predictor_path))
    action = agent.predict(obs)
    for _j in range(int(1e3)):
      obs,_r,done,_info = env.step(action)
      opt_act = _info['opt_act']
      action = agent.predict(obs,opt_act,done)

      env.render()
