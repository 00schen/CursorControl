from assistive_gym import ScratchItchJacoOracleEnv,PretrainAgent,TwoDAgent
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from stable_baselines import SAC

import numpy as np

model_path = "trained_models/ppo/ScratchItchJaco-v0.pt"
predictor_path = "../logs/sac_penalty_100/rl_model_500000_steps"

if __name__ == '__main__':
  oracle = PretrainAgent(torch.load(model_path)[0])
  env = ScratchItchJacoOracleEnv()

  env.render()
  obs = env.reset()
  agent = TwoDAgent(env,oracle,SAC.load(predictor_path))

  action = agent.predict(obs)
  while True:
    obs,_r,done,_info = env.step(action)
    action = agent.predict(obs,_info['opt_act'],done)

    env.render()
