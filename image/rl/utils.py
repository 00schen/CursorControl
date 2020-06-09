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
from stable_baselines3.common.running_mean_std import RunningMeanStd

from functools import reduce
from collections import deque
import os

from tqdm import tqdm

dirname = os.path.dirname(os.path.abspath(__file__))
parentname = os.path.dirname(dirname)
class BatchNoise():
	def __init__(self, action_space, dim, sd=.1, dropout=.3, lag=.85, batch=1, include=(0,1,2)):
		self.sd = sd
		self.batch = batch
		self.dim = dim
		self.lag = lag
		self.dropout = dropout

		self.action_space = action_space

		# random.seed(12345)
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

class Noise():
	def __init__(self, action_space, dim, sd=.1, dropout=.3, lag=.85, batch=1, include=(0,1,2)):
		self.sd = sd
		self.dim = dim
		self.lag = lag
		self.dropout = dropout

		self.action_space = action_space
		self.noises = [[self._add_grp,self._add_lag,self._add_dropout][i] for i in include]

	def reset(self):
		self.noise = random.normal(np.identity(self.dim), self.sd)
		self.lag_buffer = deque([],10)

	def _add_grp(self, action):
		return action@self.noise

	def _add_dropout(self, action):
		return np.array(action if random.random() > self.dropout else self.action_space.sample())

	def _add_lag(self, action):
		self.lag_buffer.append(action)
		return np.array(self.lag_buffer.popleft() if random.random() > self.lag else self.lag_buffer[0])

	def __call__(self,action):
		return reduce(lambda value,func: func(value), self.noises, action)

class VanillaPretrain():
	def __init__(self,config):
		actor_critic, ob_rms = torch.load(os.path.join(parentname,'trained_models','ppo',config['env0']+'.pt'))
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
	def __init__(self,config):
		self.model = SAC.load("rl_model_799985_steps")
	
	def predict(self,obs):
		return self.model.predict(obs)

class Oracle:
	def __init__(self,config):
		self.determiner = config['determiner']
		self.indices,self.pretrain = config['indices'],config['pretrain']
		self.pretrain = self.pretrain(config)
		self.noise = Noise(spaces.Box(low=-.01*np.ones(3),high=.01*np.ones(3)),3)
		self.reset()

	def reset(self):
		self.noise.reset()

	def predict(self,obs):
		action = self.pretrain.predict(obs)
		recomend = self.determiner(action)
		recomend = self.noise(recomend)
		return {'obs':(obs[self.indices[0]],obs[self.indices[1]]),'action':recomend,'real_action':action}

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
	def __init__(self,config):
		obs_size,pred_step = config['obs_size'],config['pred_step']
		self.norm = RunningMeanStd(shape=(obs_size,))
		self.data = Data(batch_size=64)
		self.pred_step = pred_step
		self.obs_history = deque([],200)
		self.model = make_train_LSTM((200-pred_step,obs_size))
		self.model.compile(Adam(lr=5e-3),'mse')
		self.pred_model = make_pred_LSTM(obs_size)
		self.save_path = config['save_path']
		self.train_iter = 0

	def predict(self,obs):
		obs = obs.flatten()
		self.obs_history.append(obs)

		sample = self._obfilt(obs)
		if not self.train_iter % 50:
			self.pred_model.load_weights(self.save_path,by_name=True)
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
			self.model.save_weights(self.save_path,save_format='h5')

	def reset(self):
		self.obs_history.clear()
		self.pred_model.load_weights(self.save_path,by_name=True)
	
	def pretrain(self,initial_batch,epochs=1):
		self._update_data(initial_batch)
		obs,target = self.data['X'],self.data['Y']
		obs = self._obfilt(obs)
		self.model.fit(obs,target,
						epochs=1,verbose=1,
						)
		self.model.save_weights(self.save_path,save_format='h5')

	"""regular LSTM"""
	def _update_data(self,X):
		obs = X[:,:-self.pred_step,:]
		target = X[:,self.pred_step:,-3:]

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

	x,state_h,state_c = LSTM(channels,stateful=True,return_state=True,name='lstm1')(episode_input)
	# model = tf.keras.Model(inputs=[episode_input],outputs=[state_c])
	recommendation_pred = Dense(3,name='dense')(x)
	model = tf.keras.Model(inputs=[episode_input],outputs=[recommendation_pred])
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
	oracle_map = {
		'target': lambda env: lambda oracle_action: env.target_pos,
		'trajectory': lambda env: env.oracle2trajectory,
	}

	def __init__(self,config):
		self.env,self.pretrain = config['env1'],config['pretrain']
		self.env,self.pretrain = self.env(),self.pretrain(config)
		self.timesteps = 0
		self.step_limit = config['step_limit']
		self.success_count = deque([0]*20,20)
		config['determiner'] = SparseEnv.oracle_map[config['oracle']](self.env)
		self.oracle = Oracle(config)

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

		if self.timesteps >= self.step_limit:
			done = True
			r = 100*(self.env.task_success > 0)
			self.success_count.append(self.env.task_success > 0)
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

default_config = {
	'pred_step': 10,
	'step_limit': 200,
	}
env_keys = ('env1','env0','obs_size','indices','pretrain')
env_map = {
		"ScratchItchJaco-v1": dict(zip(env_keys,(assistive_gym.ScratchItchJacoDirectEnv,"ScratchItchJaco-v0",27,(slice(7),slice(13,30)),VanillaPretrain))),
		"FeedingJaco-v1": dict(zip(env_keys,(assistive_gym.FeedingJacoDirectEnv,"FeedingJaco-v0",25,(slice(7),slice(10,25)),VanillaPretrain))),
		"Laptop-v1": dict(zip(env_keys,("Laptop-v1","Laptop-v0",30,(slice(30),),SBPretrain))),
	}
