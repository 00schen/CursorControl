import numpy as np
import numpy.random as random
from scipy.cluster.vq import kmeans2

import torch
import tensorflow as tf

import assistive_gym
import gym
from gym import spaces

from functools import reduce
from collections import deque
import os

from tqdm import tqdm

dirname = os.path.dirname(os.path.abspath(__file__))
parentname = os.path.dirname(dirname)

class Noise():
    def __init__(self, action_space, dim, sd=.1, dropout=.3, lag=.85, batch=1, include=(0,1,2)):
        self.sd = sd
        self.dim = dim
        self.lag = lag
        self.dropout = dropout

        self.action_space = action_space

        # random.seed(12345)
        self.noise = random.normal(np.identity(self.dim), self.sd)
        self.lag_buffer = deque([],10)

        self.noises = [[self._add_grp,self._add_lag,self._add_dropout][i] for i in include]

    def _add_grp(self, action):
        return action@self.noise

    def _add_dropout(self, action):
        return np.array(action if random.random() > self.dropout else self.action_space.sample())

    def _add_lag(self, action):
        self.lag_buffer.append(action)
        return np.array(self.lag_buffer.popleft() if random.random() > self.lag else self.lag_buffer[0])

    def __call__(self,action):
        return reduce(lambda value,func: func(value), self.noises, action)

class SampleMeanStd:
    def __init__(self,stats):
        self.mean = stats[0]
        self.sd = stats[1]

    def __call__(self,sample):
        return (sample-self.mean)/self.sd

    def inverse(self,sample):
        return sample*self.sd + self.mean

class PretrainAgent():
    def __init__(self,model_path):
        actor_critic, ob_rms = torch.load(model_path)
        self.model = actor_critic
        self.norm = ob_rms

        self.recurrent_hidden_states = [torch.zeros(1, self.model.recurrent_hidden_state_size)]
        self.masks = [torch.zeros(1, 1)]

    def add(self):
        self.recurrent_hidden_states.append(torch.zeros(1, self.model.recurrent_hidden_state_size))
        self.masks.append(torch.zeros(1, 1))
        return len(self.masks)-1

    def reset(self):
        self.recurrent_hidden_states = [torch.zeros(1, self.model.recurrent_hidden_state_size) for _i in range(len(self.recurrent_hidden_states))]
        self.masks = [torch.zeros(1,1) for _i in range(len(self.masks))]

    def predict(self,obs,done,idx=0):
        obs = self._obfilt(obs)
        obs = torch.tensor(obs,dtype=torch.float)

        self.masks[idx].fill_(0.0 if done else 1.0)
        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states[idx] = self.model.act(
                obs, self.recurrent_hidden_states[idx], self.masks[idx], deterministic=True)
        
        return action.numpy()[0,0,0]

    def _obfilt(self, obs):
        obs = np.clip((obs - self.norm.mean) / np.sqrt(self.norm.var + 1e-8), -10, 10)
        return obs

class TrajectoryOracle:


    def __init__(self,env_name,determiner):
        self.pretrain = pretrain
        self.action_space = spaces.Box(low=-1*np.ones(3),high=np.ones(3))
        self.oracle2trajectory = oracle2trajectory
        if env_name == "ScratchItchJaco-v1":
            self.indices = (slice(7),slice(13,30))
        elif env_name == "FeedingJaco-v1":
            self.indices = (slice(7),slice(10,25))

    def reset(self):
        self.noise = Noise(self.action_space,3)

    def predict(self,obs,done=False):
        action = self.pretrain.predict(obs,done,0)
        trajectory = self.oracle2trajectory(action)
        trajectory = self.noise(trajectory)
        return {'obs':(obs[self.indices[0]],obs[self.indices[1]]),'action':trajectory,'real_action':action}

class TargetOracle:
    N=3

    def __init__(self,env_name,determiner):
        self.pretrain = pretrain
        self.action_space = spaces.Box(low=-1*np.ones(3),high=np.ones(3))
        if env_name == "ScratchItchJaco-v1":
            self.indices = (slice(7),slice(13,30))
        elif env_name == "FeedingJaco-v1":
            self.indices = (slice(7),slice(10,25))
        self.env = env.get_target_pos

    def reset(self):
        self.noise = Noise(self.action_space,self.N)

    def predict(self,obs,done=False):
        action = self.pretrain.predict(obs,done,0)
        target = self.env.target_pos
        target = self.noise(target)
        return {'obs':(obs[self.indices[0]],obs[self.indices[1]]),'action':target,'real_action':action}

class SparseEnv(gym.Env):
    gym_name_mappings = {
        'ScratchItchJaco-v1': assistive_gym.ScratchItchJacoDirectEnv,
        'FeedingJaco-v1': assistive_gym.FeedingJacoDirectEnv,
    }
    oracle_mappings = {
        'target': lambda : TargetOracle(),
        'trajectory': TrajectoryOracle,
    }

    def __init__(self,env_name,step_limit=100,oracle_type='target'):
        self.env = self.gym_name_mappings[env_name]()
        self.timesteps = 0
        self.step_limit = step_limit
        self.oracle = self.oracle_mappings

    def step(self,action):
        """
        action: goal prediction
        obs: goal-conditioned observation w/o goal info + oracle recommendation
        r: sparse success/fail at end of episode
        done: only at the end of the episode
        info: {oracle recommendation alone, }
        """


    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

class BufferAgent():
    buffer_length = 10
    success_length = 5
    def __init__(self,pretrain,predictor,target2obs):      
        self.pretrain = pretrain
        self.predictor = predictor
        self.target2obs = target2obs
         
    def reset(self,model=None):
        self.prediction_buffer = deque([],self.buffer_length)
        self.pretrain.reset()
        self.predictor.reset(model)

    def predict(self,obs,done=False):
        pred_target = self.predictor.predict(np.concatenate((*obs['obs'],obs['action'])))

        self.prediction_buffer.append(pred_target)
        mean_pred = np.mean(self.prediction_buffer,axis=0)  

        pred_obs = self.target2obs(mean_pred,obs['obs'])

        action = self.pretrain.predict(pred_obs,done,1)  
        return action
