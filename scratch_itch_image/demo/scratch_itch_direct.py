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
predictor_path = os.path.join(dirname,'final_model.h5')
env_name = 'ScratchItchJacoDirect-v0'

if __name__ == '__main__':
  env = gym.make(env_name)

  agent = BufferAgent(PretrainAgent(pretrain_path),Predictor(predictor_path),env)
  oracle = TrajectoryOracle(PretrainAgent(pretrain_path),env)

  env.render()

  successes = []
  for j in range(5):
    obs = env.reset()

    oracle.reset()
    # obs = obs[0]
    obs = oracle.predict(obs)
    obs['sample'] = env.observation_space.sample()

    agent.reset()
    action = agent.predict(obs)
    for i in range(200):
      obs,_r,done,_info = env.step(action)
      # obs,done,_info = obs[0],done[0],_info[0]
      obs = oracle.predict(obs,done)
      # obs['sample'] = env.observation_space.sample()

      action = agent.predict(obs,done)

      env.render()
  env.close()
      # if i == 99:
      #   successes.append(info[0]['task_success'])
    #   print("episode finished")
    # print("round %d finished" % j)
  # print(np.mean(successes))
  # print(np.var(successes))
  # with open('predict_scratch1.csv', 'w', newline='') as myfile:
  #    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
  #    wr.writerow(successes)