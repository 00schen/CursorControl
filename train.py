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
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.callbacks import EvalCallback


time_now = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
best_model_save_path = "sac_best_%s" % time_now
log_path = "sac_%s" % time_now
os.makedirs(best_model_save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)

env = gym.make('cursorcontrol-v1')
env = Monitor(env, log_path)
eval_env = gym.make('cursorcontrol-v1')

model = SAC(MlpPolicy, env, verbose=1)
callback = EvalCallback(eval_env, best_model_save_path=best_model_save_path, log_path=log_path)

time_steps = int(1e6)

model.learn(total_timesteps=time_steps,callback=callback)
print("Training Done")

results_plotter.plot_results([log_path], time_steps, results_plotter.X_TIMESTEPS, "SAC CursorControl")
plt.show()

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()