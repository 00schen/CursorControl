import numpy as np
import numpy.random as random

import torch
import tensorflow as tf

from gym import spaces

from functools import reduce
from collections import deque
import os

class Noise():
    def __init__(self, action_space, dim, sd=.1, dropout=.3, lag=.3, batch=1,include=(0,1,2)):
        self.sd = sd
        self.batch = batch
        self.dim = dim
        self.lag = lag
        self.dropout = dropout

        self.action_space = action_space

        random.seed(12345)
        self.noise = random.normal(np.repeat(np.identity(self.dim)[np.newaxis,...],self.batch,0), self.sd)
        self.lag_buffer = [deque([],10) for i in range(self.batch)]

        self.noises = [[self._add_grp,self._add_lag,self._add_dropout][i] for i in include]

    def _add_grp(self, action):
        return (action[:,np.newaxis,:]@self.noise)[:,0,:]

    def _add_dropout(self, action):
        return np.array([action_i if random.random() > self.dropout else self.action_space.sample() for action_i in action])

    def _add_lag(self, action):
        for buffer,act in zip(self.lag_buffer,action):
            buffer.append(act)
        return np.array([buffer.popleft() if random.random() > self.lag else buffer[0] for buffer in self.lag_buffer])

    def __call__(self,action):
        return reduce(lambda value,func: func(value), self.noises, action)

class SampleMeanStd:
    def __init__(self,idx):
        dirname = os.path.dirname(__file__)
        stats_path = os.path.join(dirname,'trajectory_stats.npy')
        stats = np.load(stats_path)
        self.mean = stats[0,idx]
        self.sd = stats[1,idx]

    def __call__(self,sample):
        return (sample-self.mean)/self.sd

class Predictor:
    def __init__(self, predictor_path):
        self.model = tf.keras.models.load_model(predictor_path)
        self.norm = SampleMeanStd(list(range(7))+list(range(13,33)))

    def reset(self):
        # self.buffer = deque([[0]*31]*200,200)
        self.buffer = deque([[0]*27]*200,200)

        # idx = np.concatenate((range(7),range(13,30)))
        # self.buffer = deque([np.concatenate((env.observation_space.sample()[idx],env.action_space.sample())) for i in range(200)],200)

    def predict(self,obs):
        obs = self.norm(obs)

        # if not self.buffer:
        #   self.buffer = deque([obs]*200,200)
        # else:
        #   self.buffer.append(obs)
        self.buffer.append(obs)
        return self.model.predict(np.array(self.buffer)[np.newaxis,...])[0][-1]

class PretrainAgent():
    def __init__(self,model_path):
        actor_critic, ob_rms = torch.load(model_path)
        self.model = actor_critic
        self.norm = ob_rms

        self.recurrent_hidden_states = torch.zeros(1, self.model.recurrent_hidden_state_size)
        self.masks = torch.zeros(1, 1)

    def reset(self):
        self.recurrent_hidden_states = torch.zeros(1, self.model.recurrent_hidden_state_size)

    def predict(self,obs,done):
        obs = self._obfilt(obs)
        obs = torch.tensor(obs,dtype=torch.float)

        self.masks.fill_(0.0 if done else 1.0)
        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states = self.model.act(
                obs, self.recurrent_hidden_states, self.masks, deterministic=True)
        
        return action.numpy()[0,0,0]

    def _obfilt(self, obs):
        # self.norm.update(obs)
        obs = np.clip((obs - self.norm.mean) / np.sqrt(self.norm.var + 1e-8), -10, 10)
        return obs

class ScratchItchOracle:
    N=7

    def __init__(self,pretrain,env):
        self.pretrain = pretrain
        self.action_space = env.action_space

    def reset(self):
        self.noise = Noise(self.action_space,self.N)

    def predict(self,obs,done=False):
        action = self.noise(self.pretrain.predict(obs,done)[np.newaxis,:])[0]
        # action = self.pretrain.predict(obs,done)
        return {'obs':(obs[:7],obs[13:]),'action':action}

class TrajectoryOracle:
    N=3

    def __init__(self,pretrain,env):
        self.pretrain = pretrain
        self.action_space = spaces.Box(low=-1*np.ones(3),high=np.ones(3))
        self.oracle2trajectory = env.oracle2trajectory

    def reset(self):
        self.noise = Noise(self.action_space,self.N)

    def predict(self,obs,done=False):
        action = self.pretrain.predict(obs,done)
        trajectory = self.oracle2trajectory(action)
        trajectory = self.noise(trajectory[np.newaxis,:])[0]
        # action = self.pretrain.predict(obs,done)
        return {'obs':(obs[:7],obs[13:]),'action':trajectory}

from scipy.spatial.distance import cosine as cosine_sim

class BufferAgent():
    buffer_length = 50
    success_length = 5
    def __init__(self,pretrain,predictor,env):      
        self.pretrain = pretrain
        self.predictor = predictor
        self.target2obs = env.target2obs
         
    def reset(self):
        self.prediction_buffer = deque([],10)
        self.pretrain.reset()
        self.predictor.reset()

    def predict(self,obs,done=False):
        pred_target = self.predictor.predict(np.concatenate((*obs['obs'],obs['action'])))
        # pred_target = self.predictor.predict(np.concatenate((obs['sample'][:7],obs['sample'][13:],obs['action'])))

        self.prediction_buffer.append(pred_target)
        mean_pred = np.mean(self.prediction_buffer,axis=0)  

        pred_obs = self.target2obs(mean_pred,obs['obs'])

        action = self.pretrain.predict(pred_obs,done)  
        return action
        # return obs['action']
