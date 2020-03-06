import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import time
import os

import CursorControl

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines.results_plotter import ts2xy, load_results, plot_curves, X_EPISODES
from stable_baselines.bench import Monitor
from stable_baselines.common import callbacks

def plot_results(dirs, num_timesteps, xaxis, task_name):
    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name)
    plt.legend(list(zip(*xy_list))[0],[s.replace("sac_best_","") for s in dirs])


eval_env = gym.make('cursorcontrol-v1')
env = gym.make('cursorcontrol-v1')
time_steps = int(5e4)
gammas = [.1,.3,.5,.7]

plot_results(["sac_%f" % gamma for gamma in gammas], time_steps, X_EPISODES, "SAC CursorControl")
plt.show()

model = SAC(MlpPolicy, env, verbose=1)
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

