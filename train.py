import numpy as np
import gym
import tensorflow as tf
import time

import CursorControl

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import SAC

env = gym.make('cursorcontrol-v1')

model = SAC(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
print("Training Done")
time_now = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
model.save("sac_%s" % time_now)

obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()