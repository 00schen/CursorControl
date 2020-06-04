import assistive_gym
import gym

import numpy as np
import numpy.random as random

import os,sys

dirname = os.path.dirname(__file__)
parentname = os.path.dirname(dirname)
sys.path.append(parentname)

from stable_baselines3.sac import SAC

# from utils import *
env_name = 'LaptopJaco-v0'

if __name__ == '__main__':
  env = gym.make(env_name)

  agent = SAC.load("rl_model_1000000_steps")

  env.render()

  successes = []
  while True:
    obs = env.reset()

    for i in range(200):
      action = agent.predict(obs)[0]

      obs,_r,done,_info = env.step(action)

      env.render()
  env.close()