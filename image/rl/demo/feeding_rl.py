import gym
import time
import os,sys
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf
import torch
import numpy as np

from stable_baselines3.sac import MlpPolicy
from stable_baselines3.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks

dirname = os.path.dirname(os.path.abspath(__file__))
parentname = os.path.dirname(dirname)
sys.path.append(parentname)
from utils import MainEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env_name',help='gym environment name')
parser.add_argument('--oracle',default='trajectory',help='oracle to use')
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    sparse_kwds = {'oracle_type':args.oracle}
    env = MainEnv(args.env_name,os.path.join(dirname,'decoder_160000_steps','decoder'),pretrain_decoder=False,
                    sparse_kwds=sparse_kwds)
    norms = torch.load(os.path.join(dirname,'norm_160000_steps'))
    env.load_norms(norms['env_norm'],norms['decoder_norm'])

    model = SAC.load(os.path.join(dirname,'rl_model_160000_steps.zip'))

    env.render()
    while True:
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action = model.predict(obs)[0]
            obs,_r,done,_info = env.step(action)

