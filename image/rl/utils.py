import numpy as np
import numpy.random as random
from scipy.cluster.vq import kmeans2

import torch
import tensorflow as tf
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.optimizers import Adam

import assistive_gym
import gym
from gym import spaces
from stable_baselines3.sac import SAC
from stable_baselines3.sac import MlpPolicy
try:
    from stable_baselines3.common.running_mean_std import RunningMeanStd
except ImportError:
    from stable_baselines.common.running_mean_std import RunningMeanStd

from functools import reduce
from collections import deque
import os

from tqdm import tqdm

dirname = os.path.dirname(os.path.abspath(__file__))
parentname = os.path.dirname(dirname)

# class Noise():
#     def __init__(self, action_space, dim, sd=.1, dropout=.3, lag=.85, batch=1, include=(0,1,2)):
#         self.sd = sd
#         self.batch = batch
#         self.dim = dim
#         self.lag = lag
#         self.dropout = dropout

#         self.action_space = action_space

#         # random.seed(12345)
#         self.noise = random.normal(np.repeat(np.identity(self.dim)[np.newaxis,...],self.batch,0), self.sd)
#         self.lag_buffer = [deque([],10) for i in range(self.batch)]

#         self.noises = [[self._add_grp,self._add_lag,self._add_dropout][i] for i in include]

#     def _add_grp(self, action):
#         return (action[:,np.newaxis,:]@self.noise)[:,0,:]

#     def _add_dropout(self, action):
#         return np.array([action_i if random.random() > self.dropout else self.action_space.sample() for action_i in action])

#     def _add_lag(self, action):
#         for buffer,act in zip(self.lag_buffer,action):
#             buffer.append(act)
#         return np.array([buffer.popleft() if random.random() > self.lag else buffer[0] for buffer in self.lag_buffer])

#     def __call__(self,action):
#         return reduce(lambda value,func: func(value), self.noises, action)

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

class VanillaPretrain():
    def __init__(self,env_name):
        actor_critic, ob_rms = torch.load(os.path.join(parentname,'trained_models','ppo',env_name+'.pt'))
        self.model = actor_critic
        self.norm = ob_rms

    def predict(self,obs):
        obs = self._obfilt(obs)
        obs = torch.tensor(obs,dtype=torch.float)

        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states = self.model.act(obs,
            torch.zeros(1, self.model.recurrent_hidden_state_size), torch.ones(1,1), deterministic=True)
        
        return action.numpy()[0,0,0]

    def _obfilt(self, obs):
        obs = np.clip((obs - self.norm.mean) / np.sqrt(self.norm.var + 1e-8), -10, 10)
        return obs

class SBPretrain:
    def __init__(self,env_name):
        self.model = SAC.load("rl_model_1000000_steps")
    
    def predict(self,obs):
        return self.model.predict(obs)

class TrajectoryOracle:
    env_map = {
        "ScratchItchJaco-v1": ((slice(7),slice(13,30)),VanillaPretrain),
        "FeedingJaco-v1": ((slice(7),slice(10,25)),VanillaPretrain),
        "Laptop-v1": ((slice(30),),SBPretrain),
    }

    def __init__(self,env_name,determiner):
        self.oracle2trajectory = determiner

        env_name0 = env_name[:-1]+'0'
        env_name1 = env_name[:-1]+'1'
        self.indices,self.pretrain = TrajectoryOracle.env_map[env_name1]
        self.pretrain = self.pretrain(env_name0)
        self.reset()

    def reset(self):
        self.noise = Noise(spaces.Box(low=-.01*np.ones(3),high=.01*np.ones(3)),3)

    def predict(self,obs):
        action = self.pretrain.predict(obs)
        trajectory = self.oracle2trajectory(action)
        trajectory = self.noise(trajectory)
        return {'obs':(obs[self.indices[0]],obs[self.indices[1]]),'action':trajectory,'real_action':action}

class TargetOracle:
    env_map = {
        "ScratchItchJaco-v1": ((slice(7),slice(13,30)),VanillaPretrain),
        "FeedingJaco-v1": ((slice(7),slice(10,25)),VanillaPretrain),
        "Laptop-v1": ((slice(30),),SBPretrain),
    }

    def __init__(self,env_name,determiner):
        self.get_target_pos = determiner

        env_name0 = env_name[:-1]+'0'
        env_name1 = env_name[:-1]+'1'
        self.indices,self.pretrain = TargetOracle.env_map[env_name1]
        self.pretrain = self.pretrain(env_name0)
        self.reset()

    def reset(self):
        self.noise = Noise(spaces.Box(low=-.01*np.ones(3),high=.01*np.ones(3)),3)

    def predict(self,obs):
        action = self.pretrain.predict(obs)
        target = self.get_target_pos
        target = self.noise(target)
        return {'obs':(obs[self.indices[0]],obs[self.indices[1]]),'action':target,'real_action':action}

class Data:
    def __init__(self,pool_size=int(5e4),batch_size=256):
        self.data = {'X':deque([],pool_size),'Y':deque([],pool_size)}
        self.batch_size = batch_size

    def update(self,X,Y):
        self.data['X'].extend(X)
        self.data['Y'].extend(Y)

    def __getitem__(self,key):
        assert key == 'X' or key == 'Y', "Data only has keys 'X' and 'Y'."
        return np.array(self.data[key])

    def batch(self):
        idx = np.random.choice(range(len(self.data['X'])),self.batch_size,replace=False)
        return np.array(self.data['X'])[idx],np.array(self.data['Y'])[idx]

class Supervised:
    def __init__(self,save_path,obs_size,pred_step=10):
        self.norm = RunningMeanStd(shape=(obs_size,))
        self.data = Data(batch_size=64)
        self.pred_step = pred_step
        self.obs_history = deque([],200)
        self.model = make_train_LSTM((200-pred_step,obs_size))
        self.model.compile(Adam(lr=5e-3),'mse')
        self.pred_model = make_pred_LSTM(obs_size)
        self.save_path = save_path
        self.train_iter = 0

    def predict(self,obs):
        obs = obs.flatten()
        self.obs_history.append(obs)

        sample = self._obfilt(obs)
        if not self.train_iter % 50:
            self.pred_model.load_weights(self.save_path)
        hidden_state = self.pred_model.predict(sample.reshape((1,1,-1)))
        return np.concatenate((obs,hidden_state.flatten()))

    def train(self):
        if len(self.obs_history) == self.obs_history.maxlen:
            self._update_data(np.array(self.obs_history)[np.newaxis,...])
        obs,target = self.data.batch()
        sample = self._obfilt(obs)
        self.model.train_on_batch(sample,target)
        self.train_iter += 1
        if not self.train_iter % 50:
            self.model.save_weights(self.save_path)

    def reset(self):
        self.obs_history.clear()
    
    def pretrain(self,initial_batch,epochs=1):
        self._update_data(initial_batch)
        obs,target = self.data['X'],self.data['Y']
        obs = self._obfilt(obs)
        self.model.fit(obs,target,
                        epochs=1,verbose=1,
                        )
        self.model.save_weights(self.save_path)

    """regular LSTM"""
    def _update_data(self,X):
        obs = X[:,:X.shape[1]-self.pred_step,:]
        target = X[:,self.pred_step:X.shape[1],-3:]

        self.norm.update(X.reshape((-1,X.shape[-1])))
        self.data.update(obs,target)

    def _obfilt(self, obs):
        obs = np.clip((obs - self.norm.mean) / np.sqrt(self.norm.var + 1e-8), -10, 10)
        return obs

def make_train_LSTM(input_shape,channels=64):
    episode_input = tf.keras.Input(shape=input_shape,name='episode')

    x = LSTM(channels,return_sequences=True,name='lstm1')(episode_input)
    recommendation_pred = Dense(3,name='dense')(x)
    model = tf.keras.Model(inputs=[episode_input],outputs=[recommendation_pred])
    return model

def make_pred_LSTM(input_shape,channels=64):
    episode_input = tf.keras.Input(batch_shape=(1,1,input_shape),name='episode')

    x,state_h,state_c = LSTM(channels,stateful=True,return_state=True,name='lstm2')(episode_input)
    model = tf.keras.Model(inputs=[episode_input],outputs=[state_c])
    return model

def make_train_seq2seq(obs_shape,channels=64):
    obs_input = tf.keras.Input(shape=(None,obs_shape),name='observation')
    traj_input = tf.keras.Input(shape=(None,3),name='trajectory')
    
    x,state_h,state_c = LSTM(channels,return_sequences=True,return_state=True,name='encoder1')(obs_input)
    
    x = LSTM(channels,return_sequences=True,name='decoder1')(traj_input,initial_state=[state_h,state_c])
    
    recommendation_pred = Dense(3,name='dense')(x)
    model = tf.keras.Model(inputs=[obs_input,traj_input],outputs=[recommendation_pred])
    return model

class SparseEnv(gym.Env):
    env_map = {
        'ScratchItchJaco-v1': (assistive_gym.ScratchItchJacoDirectEnv,VanillaPretrain),
        'FeedingJaco-v1': (assistive_gym.FeedingJacoDirectEnv, VanillaPretrain),
    }
    oracle_map = {
        'target': lambda env,env_name: TargetOracle(env_name,env.get_target_pos),
        'trajectory': lambda env,env_name: TrajectoryOracle(env_name,env.oracle2trajectory),
    }

    def __init__(self,env_name,step_limit=200,oracle_type='trajectory'):
        env_name0 = env_name[:-1]+'0'
        env_name1 = env_name[:-1]+'1'
        self.env,self.pretrain = SparseEnv.env_map[env_name1]
        self.env,self.pretrain = self.env(),self.pretrain(env_name0)
        self.timesteps = 0
        self.step_limit = step_limit
        self.oracle = SparseEnv.oracle_map[oracle_type](self.env,env_name1)

    def step(self,action):
        """
        action: goal prediction
        obs: goal-conditioned observation w/o goal info + oracle recommendation
        r: sparse success/fail at end of episode
        done: only at the end of the episode
        info: {oracle recommendation alone, }
        """
        self.timesteps += 1

        pred_obs = self.env.target2obs(action,self.og_obs)
        action = self.pretrain.predict(pred_obs)
        obs,_r,_done,info = self.env.step(action)

        oracle_obs = self.oracle.predict(obs)
        self.og_obs = oracle_obs['obs']
        obs = np.concatenate((*oracle_obs['obs'],oracle_obs['action']))

        if self.timesteps == self.step_limit:
            done = True
            r = self.env.task_success > 0
        else:
            done = False
            r = 0

        info.update({
            'real_action': oracle_obs['real_action'],
            'oracle_recommendation': oracle_obs['action'],
        })

        return obs,r,done,info

    def reset(self):
        self.timesteps = 0
        self.oracle.reset()
        obs = self.env.reset()
        obs = self.oracle.predict(obs)
        self.og_obs = obs['obs']
        obs = np.concatenate((*obs['obs'],obs['action']))
        return obs

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

class CurriculumEnv(SparseEnv):
    def __init__(self,env_name,sparse_kwds={}):
        super().__init__(env_name,**sparse_kwds)

        self.episodes = 0
        self.success_count = 0
        self.radius_limits = (0.01,.5)
        self.success_radius = self.radius_limits[1]
        self.gamma = 1
        self.phi = lambda s,g: np.linalg.norm(s-g)

    def step(self,action):
        old_dist = self.phi(self.env.tool_pos,self.env.target_pos)

        obs,r,done,info = super().step(action)

        if self.success_radius > self.radius_limits[0]: # up to certain distance, count just getting nearer the target
            self.episode_success += np.linalg.norm(self.env.tool_pos-self.env.target_pos) < self.success_radius
            r += self.gamma*(old_dist - self.phi(self.env.tool_pos,self.env.target_pos))
        else: # then only count actually doing the task
            self.episode_success = self.task_success > 0

        if done:
            r += 100 if self.episode_success else 0

        return obs,r,done,info

    def reset(self):
        self.episodes += 1
        self.episode_success = 0
        if self.success_count / self.episodes > .5:
            self.success_radius *= .95

        return super().reset()


class MainEnv(CurriculumEnv):
    env_map = {
        'ScratchItchJaco-v1': (27,'noised_trajectory.npz'),
        'FeedingJaco-v1': (25,'f.noised_trajectory.npz'),
    }

    def __init__(self,env_name,decoder_save_path,data_path,pretrain_decoder=True,curriculum_kwds={},sparse_kwds={},decoder_kwds={}):
        super().__init__(env_name,sparse_kwds=sparse_kwds,**curriculum_kwds)
        obs_size,data_file = MainEnv.env_map[env_name]

        self.observation_space = spaces.Box(low=-10*np.ones(obs_size+64),high=10*np.ones(obs_size+64))
        self.action_space = spaces.Box(low=-1*np.ones(3),high=np.ones(3))
        self.norm = RunningMeanStd(shape=self.observation_space.shape)
        
        self.decoder = Supervised(decoder_save_path,obs_size,**decoder_kwds)
        if pretrain_decoder:
            X,_Y = np.load(os.path.join(data_path,data_file)).values()
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

    def _obfilt(self, obs):
        self.norm.update(obs.reshape((1,-1)))
        obs = np.clip((obs - self.norm.mean) / np.sqrt(self.norm.var + 1e-8), -10, 10)
        return obs

class HEREnv(gym.GoalEnv,MainEnv):
    def __init__(self,env_name,decoder_save_path,data_path,pretrain_decoder=False,sparse_kwds={},decoder_kwds={}):
        MainEnv.__init__(self,env_name,decoder_save_path,data_path,pretrain_decoder=pretrain_decoder,sparse_kwds=sparse_kwds,decoder_kwds=decoder_kwds)
        self.observation_space = spaces.Dict({
            'observation': self.observation_space,
            'desired_goal': spaces.Box(-1*np.ones(3), np.ones(3)),
            'achieved_goal': spaces.Box(-1*np.ones(3), np.ones(3))
            })

    def step(self,action):
        obs,r,done,info = super().step(action)
        # obs = {'observation':obs, 'desired_goal': self.env.target_pos, 'achieved_goal': }
