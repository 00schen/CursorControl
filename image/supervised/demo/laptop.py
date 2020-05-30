import assistive_gym
import gym

import numpy as np
import numpy.random as random

import os,sys

dirname = os.path.dirname(__file__)
parentname = os.path.dirname(dirname)
sys.path.append(parentname)

from utils import *
pretrain_path = os.path.join('trained_models','ppo','ScratchItchJaco-v0.pt')
env_name = 'LaptopJaco-v0'

if __name__ == '__main__':
  env = gym.make(env_name)

  agent = PretrainAgent(pretrain_path)

  env.render()

  successes = []
  for j in range(5):
    obs = env.reset()

    action = agent.predict(obs,False)
    for i in range(200):
      obs,_r,done,_info = env.step(action)

      action = agent.predict(obs,done)

      env.render()
  env.close()