import gym
import time
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf
import torch
import numpy as np

from stable_baselines3.sac import MlpPolicy
from stable_baselines3.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import callbacks

from utils import deque,RunningMeanStd,spaces,Supervised,SparseEnv

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',help='path to data')
parser.add_argument('--local_dir',help='dir to save trials')
parser.add_argument('--env_name',help='gym environment name')
parser.add_argument('--exp_name',help='experiment name')
parser.add_argument('--oracle',default='trajectory',help='oracle to use')
parser.add_argument('--radius',type=float)
args, _ = parser.parse_known_args()

class CurriculumEnv(SparseEnv):
    def __init__(self,env_name,radius_max=.3,sparse_kwds={}):
        super().__init__(env_name,**sparse_kwds)

        self.success_count = deque([0]*50,50)
        self.radius_limits = (0.01,radius_max)
        self.success_radius = self.radius_limits[1]
        self.gamma = 100
        self.phi = lambda s,g: np.linalg.norm(s-g)

    def step(self,action):
        # old_dist = self.phi(self.env.tool_pos,self.env.target_pos)
        old_dist = self.phi(action,self.env.target_pos)

        obs,r,done,info = super().step(action)
        if self.success_radius > self.radius_limits[0]: # up to certain distance, count just getting nearer the target
            # self.episode_success += np.linalg.norm(self.env.tool_pos-self.env.target_pos) < self.success_radius
            # r += self.gamma*(old_dist - self.phi(self.env.tool_pos,self.env.target_pos))
            self.episode_success += np.linalg.norm(action-self.env.target_pos) < self.success_radius
            r += self.gamma*(old_dist - self.phi(action,self.env.target_pos))
        else: # then only count actually doing the task
            self.episode_success = self.task_success > 0

        if done:
            r += 100 if self.episode_success else 0
            self.success_count.append(self.episode_success > 0)

        return obs,r,done,info

    def reset(self):
        self.episode_success = 0
        if np.mean(self.success_count) > .5:
            self.success_radius *= .95

        return super().reset()

class MainEnv(CurriculumEnv):
    env_map = {
        'ScratchItchJaco-v1': (27,'noised_trajectory.npz'),
        # 'FeedingJaco-v1': (25,'f.noised_trajectory.npz'),
        'FeedingJaco-v1': (25,'F.noised_trajectory.npz'),
    }

    def __init__(self,env_name,decoder_save_path,pretrain_decoder=False,data_path='',curriculum_kwds={},sparse_kwds={},decoder_kwds={}):
        super().__init__(env_name,sparse_kwds=sparse_kwds,**curriculum_kwds)
        obs_size,data_file = MainEnv.env_map[env_name]

        # self.observation_space = spaces.Box(low=-10*np.ones(obs_size+64),high=10*np.ones(obs_size+64))
        self.observation_space = spaces.Box(low=-10*np.ones(obs_size+3),high=10*np.ones(obs_size+3))
        self.action_space = spaces.Box(low=-1*np.ones(3),high=np.ones(3))
        self.norm = RunningMeanStd(shape=self.observation_space.shape)
        
        self.decoder = Supervised(decoder_save_path,obs_size,**decoder_kwds)
        if pretrain_decoder:
            # X,_Y = np.load(os.path.join(data_path,data_file)).values()
            X,Y,_Z,W = np.load(os.path.join(data_path,data_file)).values()
            if 'oracle_type' in sparse_kwds and 'oracle_type' == 'target':
                W = np.repeat(W[:,np.newaxis,:],200,1)
                X = np.concatenate((X[...,list(range(7))+list(range(10,25))],W),axis=2)
            else:
                X = np.concatenate((X[...,list(range(7))+list(range(10,25))],Y),axis=2)
            X = X[:1000]
            self.decoder.pretrain(X)

    def step(self,action):
        obs,r,done,info = super().step(action)
        obs = self.decoder.predict(obs)
        obs = self._obfilt(obs)

        return obs,r,done,info

    def reset(self):
        self.decoder.reset()
        obs = super().reset()
        obs = self.decoder.predict(obs)
        obs = self._obfilt(obs)
        return obs

    def load_norms(self,obs_norm,decoder_norm):
        self.norm = obs_norm
        self.decoder.norm = decoder_norm

    def _obfilt(self, obs):
        self.norm.update(obs.reshape((1,-1)))
        obs = np.clip((obs - self.norm.mean) / np.sqrt(self.norm.var + 1e-8), -10, 10)
        return obs


logdir = os.path.join(args.local_dir,args.exp_name)

curriculum_kwds = {'radius_max': args.radius}
sparse_kwds = {'oracle_type':args.oracle}
decoder_save_path = os.path.join(logdir,'decoder','decoder')
env = MainEnv(args.env_name,decoder_save_path,pretrain_decoder=True,data_path=args.data_path,
                curriculum_kwds=curriculum_kwds,sparse_kwds=sparse_kwds)
eval_env = MainEnv(args.env_name,decoder_save_path,pretrain_decoder=False,
                curriculum_kwds=curriculum_kwds,sparse_kwds=sparse_kwds)

env = Monitor(env,None,allow_early_resets=False)
tensorboard_path = os.path.join(args.local_dir,'tensorboard')
os.makedirs(tensorboard_path, exist_ok=True)
model = SAC(MlpPolicy, env, verbose=1,tensorboard_log=tensorboard_path)

class CurriculumTensorboardCallback(callbacks.BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]
        self.logger.record('curriculum/radius', env.success_radius)
        self.logger.record('curriculum/success rate', np.mean(env.success_count))
        return True
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
        CurriculumTensorboardCallback(),
        ExtendedCheckpointCallback(save_freq=int(1e4),save_path=logdir),
        callbacks.CheckpointCallback(save_freq=int(1e4), save_path=logdir),
        ])

time_steps = int(1e6)

model.learn(total_timesteps=time_steps,callback=callback,tb_log_name=args.exp_name)
