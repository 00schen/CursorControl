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

import torch

from utils import MainEnv

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',help='path to data')
parser.add_argument('--local_dir',help='dir to save trials')
parser.add_argument('--env_name',help='gym environment name')
parser.add_argument('--exp_name',help='experiment name')
parser.add_argument('--oracle',default='trajectory',help='oracle to use')
args, _ = parser.parse_known_args()

logdir = os.path.join(args.local_dir,args.exp_name)

sparse_kwds = {'oracle_type':args.oracle}
decoder_save_path = os.path.join(logdir,'decoder','decoder')
env = MainEnv(args.env_name,decoder_save_path,args.data_path,pretrain_decoder=True,sparse_kwds=sparse_kwds)
eval_env = MainEnv(args.env_name,decoder_save_path,args.data_path,sparse_kwds=sparse_kwds)

env = Monitor(env,None,allow_early_resets=False)
tensorboard_path = os.path.join(args.local_dir,'tensorboard')
os.makedirs(tensorboard_path, exist_ok=True)
model = SAC(MlpPolicy, env, verbose=1,tensorboard_log=tensorboard_path)

class ExtendedCheckpointCallback(callbacks.BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            decoder_path = os.path.join(self.save_path, f'decoder_{self.num_timesteps}_steps','decoder')
            norm_path = os.path.join(self.save_path, f'norm_{self.num_timesteps}_steps')
            self.training_env.envs[0].decoder.model.save_weights(decoder_path)
            torch.save({
                'decoder_norm':self.training_env.envs[0].decoder.norm,
                'env_norm':self.training_env.envs[0].norm,
            },norm_path)
        return True
class PredictorTrainCallback(callbacks.BaseCallback):
    def _on_step(self):
        return True
    def _on_rollout_end(self):
       self.training_env.envs[0].decoder.train()

callback = callbacks.CallbackList([
        PredictorTrainCallback(),
        ExtendedCheckpointCallback(save_freq=int(1e4),save_path=logdir),
        callbacks.CheckpointCallback(save_freq=int(1e4), save_path=logdir),
        ])

time_steps = int(1e6)

model.learn(total_timesteps=time_steps,callback=callback,tb_log_name=args.exp_name)
