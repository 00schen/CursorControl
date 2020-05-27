import numpy as np
import numpy.random as random

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

class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, batch_size=50, shuffle=True):
		'Initialization'
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.shuffle = shuffle
		self.on_epoch_end()
		self.cache = {}

	def __len__(self):
		'Denotes the number of batches per epoch'
		return len(self.list_IDs)*500//self.batch_size

	def __getitem__(self, index):
		'Generate one batch of data'
		X, Y = self.__data_generation(self.indexes[self.count:self.count+self.batch_size//10])
		self.count += self.batch_size//10

		return X, Y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs)*50)
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
		self.count = 0
		self.cache = {}

	def __data_generation(self, indices):
		'Generates data containing batch_size samples'
		X = np.empty((self.batch_size, 201, 27))
		Y = np.empty((self.batch_size, 200, 3))

		for index,count in zip(indices,range(0,len(indices*10),10)):
			file_id,block = index//50, index%50
			if file_id in self.cache:
				X[count:count+10] = self.cache[file_id][block*10:(block+1)*10]
			else:
				self.cache[file_id] = np.load(self.list_IDs[file_id])
				X[count:count+10] = self.cache[file_id][block*10:(block+1)*10]
		if len(self.cache) > 100: 
			self.cache.popitem()
		X = X[:,:-1,:]
		Y = np.tile(np.expand_dims(X[:,-1,:3],axis=1),(1,200,1))

		return X, Y

class Data:
    def __init__(self,X,Y,split):
        idx = np.random.choice(range(len(X)),int(split*len(X)),replace=False)
        X_train,Y_train = np.delete(X,idx,axis=0),np.delete(Y,idx,axis=0)
        X_valid,Y_valid = X[idx],Y[idx]
        train,valid = {'X':X_train,'Y':Y_train},{'X':X_valid,'Y':Y_valid}
        self.data = {'train':train,'valid':valid}
        self.initial_sizes = {'train':Y_train.shape[0],'valid':Y_valid.shape[0]}
        self.split = split
    
    def reset(self):
        for data,idx in zip(self.data.values(),self.initial_sizes.values()):
            data['X'] = data['X'][:idx]
            data['Y'] = data['Y'][:idx]

    def update(self,X,Y):
        idx = np.random.choice(range(len(X)),int(self.split*len(X)),replace=False)
        print(idx.shape)
        X_train,Y_train = np.delete(X,idx,axis=0),np.delete(Y,idx,axis=0)
        X_valid,Y_valid = X[idx],Y[idx]
        print(X.shape)
        train,valid = {'X':X_train,'Y':Y_train},{'X':X_valid,'Y':Y_valid}
        for old_data,new_data in zip(self.data.values(),[train,valid]):
            old_data['X'] = np.vstack((old_data['X'],new_data['X']))
            old_data['Y'] = np.vstack((old_data['Y'],new_data['Y']))

    def __getitem__(self,key):
        assert key == 'train' or key == 'valid', "Data only has keys 'train' and 'valid'."
        return self.data[key]

# def serve_data(file,input_shape,batch_size,buffer_size):
def serve_data(file,size):
    # tf.compat.v1.disable_eager_execution()
    # tf.config.optimizer.set_jit(True)

    data = np.load(file)
    X,Y = data.values()
    X,Y = X[:size],Y[:size]
    
    X = (X-np.mean(X,axis=(0,1)))/np.std(X,axis=(0,1))
    X = X[:,:100,:]

    # interval_starts = np.random.randint(0,X.shape[1]-input_shape[0],X.shape[0])
    # for i,(sample,start) in enumerate(zip(X,interval_starts)):
    #     X[i,:input_shape[0]] = sample[start:start+input_shape[0],:]
    # X = X[:,:input_shape[0],:]

    target_mean,target_std = np.mean(Y,axis=0),np.std(Y,axis=0)
    Y = (Y-target_mean)/target_std
    # Y = np.repeat(Y[:,np.newaxis,:],input_shape[0],1)
    Y = np.repeat(Y[:,np.newaxis,:],100,1)

    # idx = np.random.choice(range(len(X)),int(.2*len(X)),replace=False)
    # X_train,Y_train = np.delete(X,idx,axis=0),np.delete(Y,idx,axis=0)
    # X_valid,Y_valid = X[idx],Y[idx]
    # Y_valid1 = Y_valid*target_std + target_mean

    # train_data = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
    # train_data = train_data.cache().shuffle(buffer_size).batch(batch_size)

    # val_data = tf.data.Dataset.from_tensor_slices((X_valid,Y_valid))
    # val_data = val_data.batch(batch_size)
    # val_data1 = tf.data.Dataset.from_tensor_slices((X_valid,Y_valid1))
    # val_data1 = val_data1.batch(batch_size)

    # return train_data,val_data
    return X,Y

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

class Predictor:
    def __init__(self,env_name,predictor=None,custom_objects=None):
        if isinstance(predictor,str):
            if custom_objects != None:
                self.model = tf.keras.models.load_model(predictor,custom_objects=custom_objects)
            else:
                self.model = tf.keras.models.load_model(predictor)
        else:
            self.model = predictor

        if env_name == "ScratchItchJaco-v1":
            self.obs_size = 27
            env = 'scratch_itch'
        elif env_name == "FeedingJaco-v1":
            self.obs_size = 25
            env = 'feeding'
        elif env_name == "twin":
            self.obs_size = 27
            env = 'twin'

        obs_stats,target_stats = np.load(os.path.join(dirname,'data',env,'stats.npz')).values()
        self.norm = SampleMeanStd(obs_stats)
        self.target_norm = SampleMeanStd(target_stats)

    def reset(self,model=None):
        self.buffer = deque([[0]*self.obs_size]*100,100)
        if model is not None:
            self.model = model

    def predict(self,obs):
        obs = self.norm(obs)

        self.buffer.append(obs)
        target = self.model.predict(np.array(self.buffer)[np.newaxis,...])[0][-1]
        target = self.target_norm.inverse(target)
        return target

class PretrainAgent():
    def __init__(self,model_path):
        actor_critic, ob_rms = torch.load(model_path)
        self.model = actor_critic
        self.norm = ob_rms

        self.recurrent_hidden_states = [torch.zeros(1, self.model.recurrent_hidden_state_size)]
        self.masks = torch.zeros(1, 1)

    def add(self):
        self.recurrent_hidden_states.append(torch.zeros(1, self.model.recurrent_hidden_state_size))

    def reset(self):
        self.recurrent_hidden_states = [torch.zeros(1, self.model.recurrent_hidden_state_size) for _i in range(len(self.recurrent_hidden_states))]

    def predict(self,obs,done,idx=0):
        obs = self._obfilt(obs)
        obs = torch.tensor(obs,dtype=torch.float)

        self.masks.fill_(0.0 if done else 1.0)
        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states[idx] = self.model.act(
                obs, self.recurrent_hidden_states[idx], self.masks, deterministic=True)
        
        return action.numpy()[0,0,0]

    def _obfilt(self, obs):
        obs = np.clip((obs - self.norm.mean) / np.sqrt(self.norm.var + 1e-8), -10, 10)
        return obs

class ActuatorOracle:
    N=7

    def __init__(self,pretrain,action_space):
        self.pretrain = pretrain
        self.action_space = action_space

    def reset(self):
        self.noise = Noise(self.action_space,self.N)

    def predict(self,obs,done=False):
        # action = self.noise(self.pretrain.predict(obs,done)[np.newaxis,:])[0]
        action = self.noise(self.pretrain.predict(obs,done,0))
        return {'obs':(obs[:7],obs[13:]),'action':action}

class TrajectoryOracle:
    N=3

    def __init__(self,pretrain,oracle2trajectory,env_name):
        self.pretrain = pretrain
        self.action_space = spaces.Box(low=-1*np.ones(3),high=np.ones(3))
        self.oracle2trajectory = oracle2trajectory
        if env_name == "ScratchItchJaco-v1":
            self.indices = (slice(7),slice(13,30))
        elif env_name == "FeedingJaco-v1":
            self.indices = (slice(7),slice(10,25))

    def reset(self):
        self.noise = Noise(self.action_space,self.N)

    def predict(self,obs,done=False):
        action = self.pretrain.predict(obs,done,0)
        trajectory = self.oracle2trajectory(action)
        # trajectory = self.noise(trajectory[np.newaxis,:])[0]
        trajectory = self.noise(trajectory)
        return {'obs':(obs[self.indices[0]],obs[self.indices[1]]),'action':trajectory,'real_action':action}

class TargetOracle:
    N=3

    def __init__(self,pretrain,env,env_name):
        self.pretrain = pretrain
        self.action_space = spaces.Box(low=-1*np.ones(3),high=np.ones(3))
        if env_name == "ScratchItchJaco-v1":
            self.indices = (slice(7),slice(13,30))
        elif env_name == "FeedingJaco-v1":
            self.indices = (slice(7),slice(10,25))
        self.env = env

    def reset(self):
        self.noise = Noise(self.action_space,self.N)

    def predict(self,obs,done=False):
        action = self.pretrain.predict(obs,done,0)
        target = self.env.target_pos
        target = self.noise(target)
        return {'obs':(obs[self.indices[0]],obs[self.indices[1]]),'action':target,'real_action':action}

class BufferAgent():
    buffer_length = 50
    success_length = 5
    def __init__(self,pretrain,predictor,target2obs):      
        self.pretrain = pretrain
        self.predictor = predictor
        self.target2obs = target2obs
         
    def reset(self,model=None):
        self.prediction_buffer = deque([],10)
        self.pretrain.reset()
        self.predictor.reset(model)

    def predict(self,obs,done=False):
        pred_target = self.predictor.predict(np.concatenate((*obs['obs'],obs['action'])))

        self.prediction_buffer.append(pred_target)
        mean_pred = np.mean(self.prediction_buffer,axis=0)  

        pred_obs = self.target2obs(mean_pred,obs['obs'])

        action = self.pretrain.predict(pred_obs,done,1)  
        return action
