import matplotlib.pyplot as plt
import gym
import time
import os
import tensorflow as tf

import CursorControl

from stable_baselines.sac import MlpPolicy
from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter
from stable_baselines.her import HERGoalEnvWrapper

# time_now = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
log_path = "../logs/her_%s" % 'pretrain_2d_obstacle'
os.makedirs(log_path, exist_ok=True)

env_name = 'velocitycontrol-v2'

gamma = .9
penalty = 20
rollout = 3

params = {'gamma':gamma,'penalty':penalty,'rollout':rollout}
env = HERGoalEnvWrapper(gym.make(env_name,**params))

class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[64]*5,
                                           layer_norm=False,
                                           feature_extraction="mlp")

time_steps = int(1e6)

# for r in range(time_steps-int(5e4),time_steps+int(5e4),int(5e4)):
#   real_log_path = log_path+"/rl_model_%d_steps"%r
#   model = SAC.load(real_log_path,env)

#   for _ in range(5):
#     obs = env.reset()
#     for i in range(100):
#       action, _states = model.predict(obs)
#       obs, rewards, done, info = env.step(action)
#       env.render(str(r))
#       if done:
#         break

real_log_path = log_path+"/best_model"
model = SAC.load(real_log_path,env)

for _ in range(5):
    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render("best")
        if done:
            break

env.close()