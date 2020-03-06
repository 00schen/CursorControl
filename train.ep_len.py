import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import time
import os

import CursorControl

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.common import callbacks


eval_env = gym.make('cursorcontrol-v1')
time_steps = int(5e4)
max_ep_lens = [50,100,200,500]

for max_ep_len in max_ep_lens:
  best_model_save_path = "sac_best_%f" % max_ep_len
  log_path = "sac_%f" % max_ep_len
  os.makedirs(best_model_save_path, exist_ok=True)
  os.makedirs(log_path, exist_ok=True)

  env = gym.make('cursorcontrol-v1')
  env.set_max_ep_len(max_ep_len)
  env = Monitor(env, log_path)

  model = SAC(MlpPolicy, env, verbose=1)
  callback = callbacks.EvalCallback(eval_env, best_model_save_path=best_model_save_path, log_path=log_path)

  model.learn(total_timesteps=time_steps,callback=callback)
  print("Training Done")

results_plotter.plot_results(["sac_%f" % max_ep_len for max_ep_len in max_ep_lens], time_steps, results_plotter.X_EPISODES, "SAC CursorControl")
plt.show()

for max_ep_len in max_ep_lens:
  env = gym.make('cursorcontrol-v1')
  model.load("sac_best_%f/best_model" % max_ep_len)
  obs = env.reset()
  for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render("max_ep_len: %f"% max_ep_len)
    if done:
      break

env.close()