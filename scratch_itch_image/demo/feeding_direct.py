import assistive_gym
import gym

import numpy as np
import numpy.random as random

import os,sys

from tqdm import tqdm
import argparse

dirname = os.path.dirname(os.path.abspath(__file__))
parentname = os.path.dirname(dirname)
sys.path.append(parentname)

from utils import *
pretrain_path = os.path.join('trained_models','ppo','FeedingJaco-v0.pt')
predictor_path = os.path.join('data','feeding','model_100.h5')
env_name = 'FeedingJacoDirect-v0'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seed for environment")
args, _ = parser.parse_known_args()

if __name__ == '__main__':
  env = gym.make(env_name)
  env.seed(int(args.seed))

  agent = BufferAgent(PretrainAgent(pretrain_path),Predictor(env_name,predictor_path),env)
  oracle = TrajectoryOracle(PretrainAgent(pretrain_path),env,env_name)

  # env.render()

  target = []
  prediction = []
  for i in tqdm(range(25)):
    obs = env.reset()

    # data collection
    target.append(env.target_pos)
    prediction.append([])

    oracle.reset()
    obs = oracle.predict(obs)

    agent.reset()
    action = agent.predict(obs)

    # data collection
    prediction[-1].append(np.mean(agent.prediction_buffer,axis=0))

    for i in range(200):
      obs,_r,done,_info = env.step(action)
      obs = oracle.predict(obs,done)

      action = agent.predict(obs,done)

      # data collection
      prediction[-1].append(np.mean(agent.prediction_buffer,axis=0))

  #     env.render()
  env.close()

  np.savez_compressed(f'feeding_target_results_{args.seed}',prediction=prediction,target=target)