import gym
import time
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf

from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.ppo import PPO
from stable_baselines3.common import callbacks
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize,SubprocVecEnv

import assistive_gym
from assistive_gym import LaptopJacoEnv

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir',help='dir to save trials')
parser.add_argument('--env_name',help='gym environment name')
parser.add_argument('--exp_name',help='experiment name')
args, _ = parser.parse_known_args()

logdir = os.path.join(args.local_dir,args.exp_name)

env_map = {
    'LaptopJaco-v0': LaptopJacoEnv,
}

if __name__ == "__main__":
    eval_env = env_map[args.env_name](200)
    env = make_vec_env(env_map[args.env_name],16,monitor_dir=logdir,env_kwargs={"time_limit":200},vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env,norm_reward=False)

    tensorboard_path = os.path.join(args.local_dir,'tensorboard')
    os.makedirs(tensorboard_path, exist_ok=True)
    model = PPO(MlpPolicy, env, verbose=1,tensorboard_log=tensorboard_path)

    callback = callbacks.CallbackList([
            callbacks.CheckpointCallback(save_freq=int(1e4), save_path=logdir),
            ])

    time_steps = int(5e6)

    model.learn(total_timesteps=time_steps,callback=callback,tb_log_name=args.exp_name)
    env.save_running_average(logdir)