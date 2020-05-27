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

env_name = 'ScratchItchJaco-v1'
env_name0 = env_name[:-1]+'0'
env_name1 = env_name[:-1]+'1'
pretrain_path = os.path.join(dirname,'trained_models','ppo',env_name0+'.pt')
predictor_path = os.path.join(parentname,'model_dagger2.h5')


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1000, help="seed for environment")
args, _ = parser.parse_known_args()

if __name__ == '__main__':
  env = gym.make(env_name)
  env.seed(int(args.seed))

  pretrain = PretrainAgent(pretrain_path)
  oracle = TrajectoryOracle(pretrain,env.oracle2trajectory,env_name1)
  pretrain.add()
  agent = BufferAgent(pretrain,Predictor('twin',predictor_path),env.target2obs)

  # env.render()

  target = []
  prediction = []
  for i in tqdm(range(25)):
    obs = env.reset()

    # data collection
    target.append(env.target_pos)
    prediction.append([])

    oracle.reset()
    obs = env.observation_space.sample()
    obs = oracle.predict(obs)
    # obs['action'] = np.random.random((3,))

    agent.reset()
    action = agent.predict(obs)
    # action = obs['real_action']

    # data collection
    prediction[-1].append(np.mean(agent.prediction_buffer,axis=0))

    for i in range(100):
      obs,_r,done,_info = env.step(action)
      obs = obs
      obs = oracle.predict(obs,done)
      # obs['action'] = np.random.random((3,))


      action = agent.predict(obs,done)
      # action = obs['real_action']

      # data collection
      prediction[-1].append(np.mean(agent.prediction_buffer,axis=0))

  #     env.render()
  env.close()

  np.savez_compressed(os.path.join(parentname,'test',f'dagger2_target_results_{args.seed}'),prediction=prediction,target=target)