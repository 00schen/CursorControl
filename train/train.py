import matplotlib.pyplot as plt
import gym
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf

import CursorControl

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.common import callbacks

# time_now = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
log_path = "../logs/sac_%s" % 'reward_gamma'
os.makedirs(log_path, exist_ok=True)

env = gym.make('cursorcontrol-v1')
eval_env = gym.make('cursorcontrol-v1')

noise = 0
gamma = .9

env.set_gamma(gamma)
eval_env.set_gamma(gamma)
env.set_oracle_noise(noise)
eval_env.set_oracle_noise(noise)
env = Monitor(env, log_path)
model = SAC(MlpPolicy, env, gamma=gamma, verbose=1)
callback = callbacks.EvalCallback(eval_env, best_model_save_path=log_path, log_path=log_path)

time_steps = int(2e4)

model.learn(total_timesteps=time_steps,callback=callback)
print("Training Done")

results_plotter.plot_results([log_path], time_steps, results_plotter.X_EPISODES, "SAC CursorControl")
plt.show()

model = SAC.load(log_path+"/best_model",env)
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()