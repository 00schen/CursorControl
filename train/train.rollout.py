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

log_path = "../logs/sac_rollout_no"
os.makedirs(log_path, exist_ok=True)


time_steps = int(2e4)
rollouts = [3,5,7,10]
noise,gamma = .00,.9

for rollout in rollouts:
        real_log_path = log_path + "_%f"%rollout

        env = gym.make('cursorcontrol-v2')
        eval_env = gym.make('cursorcontrol-v2')

        env.set_rollout(rollout)
        eval_env.set_rollout(rollout)
        env.set_gamma(gamma)
        eval_env.set_gamma(gamma)
        env.set_oracle_noise(noise)
        eval_env.set_oracle_noise(noise)
        env = Monitor(env, real_log_path)

        model = SAC(MlpPolicy, env, gamma=gamma, verbose=1)
        callback = callbacks.EvalCallback(eval_env, best_model_save_path=real_log_path, log_path=real_log_path)
        model.learn(total_timesteps=time_steps,callback=callback)
print("Training Done")