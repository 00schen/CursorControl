import matplotlib.pyplot as plt
import gym
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf

import CursorControl

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC, HER
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.common import callbacks
from stable_baselines.her import HERGoalEnvWrapper

# time_now = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
log_path = "../logs/sac_%s" % 'pretrain_2d_obstacle'
os.makedirs(log_path, exist_ok=True)

env_name = 'velocitycontrol-v2'

gamma = .9
penalty = 20
rollout = 3

params = {'gamma':gamma,'penalty':penalty,'rollout':rollout}
env = gym.make(env_name,**params)
eval_env = HERGoalEnvWrapper(gym.make(env_name,**params))
env = Monitor(env, log_path)

class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[64]*5,
                                           layer_norm=False,
                                           feature_extraction="mlp")

model = HER(CustomSACPolicy, env, SAC, verbose=1, **{'gamma': gamma})
callback = callbacks.CallbackList([
        callbacks.EvalCallback(eval_env, best_model_save_path=log_path, log_path=log_path),
        callbacks.CheckpointCallback(save_freq=int(5e4), save_path=log_path)])

time_steps = int(1e6)

model.learn(total_timesteps=time_steps,callback=callback)
print("Training Done")

# results_plotter.plot_results([log_path], time_steps, results_plotter.X_EPISODES, "SAC GoalControl")
# plt.show()