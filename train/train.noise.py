import matplotlib.pyplot as plt
import gym
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf

import CursorControl
from proxy_draw_curve import make_plot_curves

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.common import callbacks

log_path = "../logs/sac_noise"
os.makedirs(log_path, exist_ok=True)

env = gym.make('cursorcontrol-v1')
eval_env = gym.make('cursorcontrol-v1')

time_steps = int(2e4)
noises = [0,5e-2,1e-1]
gamma = .9

for noise in noises:
        real_log_path = log_path + "_%f"%noise
        env.set_gamma(gamma)
        eval_env.set_gamma(gamma)
        env.set_oracle_noise(noise)
        eval_env.set_oracle_noise(noise)
        env = Monitor(env, real_log_path)
        model = SAC(MlpPolicy, env, gamma=gamma, verbose=1)
        callback = callbacks.EvalCallback(eval_env, best_model_save_path=real_log_path, log_path=real_log_path)
        
        model.learn(total_timesteps=time_steps,callback=callback)
print("Training Done")