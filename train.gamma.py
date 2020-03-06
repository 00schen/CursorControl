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
time_steps = int(1e4)
gammas = [.1,.3,.5,.7]

for gamma in gammas:
  best_model_save_path = "sac_best_%f" % gamma
  log_path = "sac_%f" % gamma
  os.makedirs(best_model_save_path, exist_ok=True)
  os.makedirs(log_path, exist_ok=True)

  env = gym.make('cursorcontrol-v1')
  env = Monitor(env, log_path)

  model = SAC(MlpPolicy, env, verbose=1)
  callback = callbacks.EvalCallback(eval_env, best_model_save_path=best_model_save_path, log_path=log_path)

  model.learn(total_timesteps=time_steps,callback=callback)
  print("Training Done")

results_plotter.plot_results(["sac_%f" % gamma for gamma in gammas], time_steps, results_plotter.X_TIMESTEPS, "SAC CursorControl")
plt.show()

for gamma in gammas:
  env = gym.make('cursorcontrol-v1')
  model.load("sac_best_%f/best_model" % gamma)
  obs = env.reset()
  for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render("gamma: %f"% gamma)
    if done:
      break

env.close()