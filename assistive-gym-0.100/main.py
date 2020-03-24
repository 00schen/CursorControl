from assistive_gym import ScratchItchJacoOracleEnv,PretrainOracle,TwoDAgent
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from stable_baselines import SAC

import numpy as np

model_path = "trained_models/ppo/ScratchItchJaco-v0.pt"
predictor_path = "../logs/sac_penalty_100/rl_model_500000_steps"

if __name__ == '__main__':
  actor_critic, ob_rms = torch.load(model_path)
  oracle = PretrainOracle(actor_critic)

  env = make_vec_envs('ScratchItchJacoOracle-v0', 1001, 1, None, None,
                    False, device='cpu', allow_early_resets=False)

  vec_norm = get_vec_normalize(env)
  if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

  # Get a render function
  render_func = get_render_func(env)

  if render_func is not None:
    render_func('human')

  obs = env.reset()
  agent = TwoDAgent(env,oracle,SAC.load(predictor_path))

  action = agent.predict(obs)
  while True:
    obs, r, done, debug = env.step(action)
    debug = debug[0]
    action = agent.predict(obs,debug["obs_2d"],debug["target_func"],done)

    if render_func is not None:
      render_func('human')

