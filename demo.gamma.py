import matplotlib.pyplot as plt
import gym
import tensorflow as tf
import time
import os

import CursorControl
from proxy_draw_curve import make_plot_curves

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.common import callbacks

eval_env = gym.make('cursorcontrol-v1')
env = gym.make('cursorcontrol-v1')
time_steps = int(5e4)
gammas = [.1,.3,.5,.7]

results_plotter.plot_curves = make_plot_curves(gammas)
results_plotter.plot_results(["sac_%f" % gamma for gamma in gammas], time_steps, results_plotter.X_EPISODES, "SAC CursorControl")
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

