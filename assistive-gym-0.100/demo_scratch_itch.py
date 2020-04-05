from assistive_gym import ScratchItchJacoOracleEnv,PretrainAgent,TwoDAgent
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

from stable_baselines import SAC

import numpy as np

import numpy.random as random

import csv

model_path = 'trained_models/ppo/ScratchItchJaco-v0.pt'
predictor_path = '../logs/sac_pred_goal2/rl_model_500000_steps'
env_name = 'ScratchItchJacoOracle-v0'

if __name__ == '__main__':
  def set_episode_length(env):
    print("set episode called")
    env._max_episode_steps = 4000
  env = make_vec_envs(env_name, 1002, 1, None, None,
                      add_timestep=False, device='cpu', allow_early_resets=False,
                      setup_function=set_episode_length)

  # Get a render function
  render_func = get_render_func(env)

  # We need to use the same statistics for normalization as used in training
  actor_critic, ob_rms = torch.load(model_path)
  pretrain, predictor = PretrainAgent(actor_critic),SAC.load(predictor_path)

  vec_norm = get_vec_normalize(env)
  if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

  # if render_func is not None:
  #   render_func('human')

  successes = []
  for j in range(100):
    obs = env.reset()
    agent = TwoDAgent(env,pretrain,predictor)
    action = agent.predict(obs)
    for i in range(100):
      obs,_r,done,info = env.step(action)
      action = agent.predict(obs,info[0],done=done[0])

      if i == 99:
        successes.append(info[0]['task_success'])
      # if render_func is not None:
      #   render_func('human')
      print("episode finished")
    print("round %d finished" % j)
  print(np.mean(successes))
  print(np.var(successes))
  with open('predict_scratch1.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(successes)