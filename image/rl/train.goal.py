import gym
import time
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf

from stable_baselines3.sac import MlpPolicy
from stable_baselines3.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks

from assistive_gym import LaptopJacoEnv

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir',help='dir to save trials')
parser.add_argument('--env_name',help='gym environment name')
parser.add_argument('--exp_name',help='experiment name')
args, _ = parser.parse_known_args()

logdir = os.path.join(args.local_dir,args.exp_name)

env = LaptopJacoEnv(200)
eval_env = LaptopJacoEnv(200)

env = Monitor(env,None,allow_early_resets=False)
tensorboard_path = os.path.join(args.local_dir,'tensorboard')
os.makedirs(tensorboard_path, exist_ok=True)
os.makedirs(os.path.join(args.exp_name), exist_ok=True)
model = SAC(MlpPolicy, env, verbose=1,tensorboard_log=tensorboard_path)

callback = callbacks.CallbackList([
        callbacks.CheckpointCallback(save_freq=int(1e4), save_path=logdir),
        ])

time_steps = int(1e6)

model.learn(total_timesteps=time_steps,callback=callback,tb_log_name=args.exp_name)