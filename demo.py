import numpy as np
import gym
import tensorflow as tf
import time

import CursorControl

from stable_baselines.sac import MlpPolicy
from stable_baselines import SAC

env = gym.make('cursorcontrol-v1')

model = SAC(MlpPolicy, env, verbose=1)
model.load("%s" % sys.argv[1])
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break

env.close()