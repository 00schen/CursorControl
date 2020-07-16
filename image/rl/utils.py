import numpy as np

import torch as th
from tensorflow import keras
from tensorflow.keras.layers import Dense

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.callbacks import BaseCallback,CallbackList,CheckpointCallback
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common import logger

import pybullet as p

import os
import pickle

from envs import *
from collections import namedtuple
from tqdm import tqdm,trange

"""Stable Baseline Callbacks"""
class TensorboardCallback(BaseCallback):
	def __init__(self, curriculum=False, verbose=0):
		super().__init__(verbose)
		self.curriculum = curriculum
	def _on_step(self):
		base_env = self.training_env.envs[0]
		env = self.training_env.envs[0].env.env
		self.min_dist = np.minimum(self.min_dist,np.linalg.norm(env.target_pos - env.tool_pos))	
		self.logger.record('success_metric/success_rate', np.mean(base_env.success_count))
		if self.curriculum:
			self.logger.record('success_metric/t', base_env.scheduler.t)
		self.logger.record('success_metric/init_distance', np.linalg.norm(env.target_pos-env.init_pos))
		self.logger.record('success_metric/min_distance', self.min_dist)
		self.logger.record('success_metric/noop_rate', np.mean(base_env.noop_buffer))
		self.logger.record('success_metric/accuracy', np.mean(base_env.accuracy))
		# self.logger.record('success_metric/log_loss', np.mean(base_env.log_loss))

		return True
	def _on_rollout_start(self):
		self.min_dist = np.inf

class NormCheckpointCallback(BaseCallback):
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
			path = os.path.join(self.save_path, f'norm_{self.num_timesteps}_steps')
			self.training_env.save(path)
		return True

class PostProcessingReplayBuffer(ReplayBuffer):
	def reset_sample(self):
		self.sample_batch = {'obs':[], 'next_obs':[], 'action':[], 'reward':[], 'done':[], 'info':[]}
	def add(self, obs, next_obs, action, reward, done, info):
		self.sample_batch['obs'].append(obs)
		self.sample_batch['next_obs'].append(next_obs)
		self.sample_batch['action'].append(action)
		self.sample_batch['reward'].append(reward)
		self.sample_batch['done'].append(done)
		self.sample_batch['info'].append(info)
	def extend(self, *args, **kwargs):
		for data in zip(*args):
			ReplayBuffer.add(self,*data)
	def confirm(self):
		self.extend(self.sample_batch['obs'],
					self.sample_batch['next_obs'],
					self.sample_batch['action'],
					self.sample_batch['reward'],
					self.sample_batch['done'],
					self.sample_batch['info'])

class RewardRolloutCallback(BaseCallback):
	def __init__(self, relabel_all=False, verbose=0):
		super().__init__(verbose)
		self.relabel_all = relabel_all
	def _on_step(self):
		return True
	def _on_rollout_start(self):
		self.model.replay_buffer.reset_sample()
	def _on_rollout_end(self):
		batch = self.model.replay_buffer.sample_batch
		if not self.relabel_all:
			if batch['info'][-1][0]['task_success'] > 0:
				batch['reward'] += np.array([[info_[0]['diff_distance']] for info_ in batch['info']])
		else:
			batch['reward'] += np.array([[info_[0]['diff_distance']] for info_ in batch['info']])
	
		self.model.replay_buffer.confirm()

"""Ray Helpers"""
# from ray.rllib.agents import sac
# class CustomTrainable(sac.SACTrainer):
# 	def _save(self, checkpoint_dir):
# 		checkpoint_path = os.path.join(checkpoint_dir,
# 									f"checkpoint-{self.iteration}.h5")
# 		self.get_policy().model.action_model.save_weights(checkpoint_path,save_format='h5')

# 		os.makedirs(os.path.join(checkpoint_dir,"norm"), exist_ok=True)
# 		self.workers.local_worker().env.save(os.path.join(checkpoint_dir,"norm","norm"))
# 		return checkpoint_path

class RayPolicy:
	def __init__(self,obs_size,action_size,layer_sizes,path):
		model = self.model = keras.models.Sequential()
		model.add(Dense(layer_sizes[0],input_shape=(obs_size,)))
		for size in layer_sizes[1:]:
			model.add(Dense(size))
		model.add(Dense(action_size*2))
		model.load_weights(path)

	def predict(self,obs):
		obs = obs.reshape((1,-1))
		stats = self.model.predict(obs)[0]
		mean, log_std = np.split(stats, 2)
		log_std = np.clip(log_std, -20, 2)
		std = np.exp(log_std)
		action = np.random.normal(mean, std)
		return np.clip(np.tanh(action), -1, 1)

def norm_factory(base):
	class NormEnv(base):
		def __init__(self,**kwargs):
			super().__init__(**kwargs)
			self.norm = RunningMeanStd(shape=self.observation_space.shape)

		def step(self,action):
			obs,r,done,info = super().step(action)
			self.norm.update(obs[np.newaxis,...])
			obs = self._obfilt(obs)
			return obs,r,done,info

		def reset(self):
			obs = super().reset()
			self.norm.update(obs[np.newaxis,...])
			obs = self._obfilt(obs)
			return obs

		def _obfilt(self,obs):
			return (obs-self.norm.mean)/(1e-8+np.sqrt(self.norm.var))
		def save(self,path):
			with open(path, "wb") as file_handler:
				pickle.dump(self.norm, file_handler)
		def load(self,path):
			with open(path, "rb") as file_handler:
				self.norm = pickle.load(file_handler)
	return NormEnv

"""Pretrain"""
class TransitionFeeder:
	def __init__(self,config):
		obs_data,r_data,done_data,targets_data = config['obs_data'],config['r_data'],config['done_data'],config['targets_data']
		self.data = zip(obs_data,r_data,done_data,targets_data)
		self.env = config['env'].env.env
		self.env.reset()
		self.action_space = config['env'].action_space
		self.blank_obs = config['blank_obs']
	def reset(self):
		obs,r,done,targets = next(self.data,([],[],[],[]))
		self.obs,self.r,self.done,self.targets = iter(obs),iter(r),iter(done),iter(targets)
		return next(self.obs,None)
	def step(self,action):
		obs,r,done = next(self.obs,None),next(self.r,0),next(self.done,True)
		self.env.targets = next(self.targets,None)
		info = {'noop':not np.count_nonzero(obs[-6:])}

		if self.blank_obs:
			obs[:-6] = 0

		return obs,r,done,info

IndexReplayBufferSamples = namedtuple("IndexReplayBufferSamples",
	["observations","actions","next_observations", "dones", "rewards", "indices"])
class IndexReplayBuffer(ReplayBuffer):
	def __init__(self, buffer_size, observation_space, action_space, device = 'cpu', n_envs = 1):
		super().__init__(buffer_size, observation_space, action_space, device, n_envs)
		self.indices = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)		
	def add(self, obs, next_obs, action, reward, done, index=None):
		if index is not None:
			self.indices[self.pos] = np.array(index).copy()
		super().add(obs, next_obs, action, reward, done)
	def _get_samples(self,batch_inds,env):
		data = (self._normalize_obs(self.observations[batch_inds, 0, :], env),
				self.actions[batch_inds, 0, :],
				self._normalize_obs(self.next_observations[batch_inds, 0, :], env),
				self.dones[batch_inds],
				self._normalize_reward(self.rewards[batch_inds], env),
				self.indices[batch_inds])
		return IndexReplayBufferSamples(*tuple(map(self.to_torch, data)))

def pretrain(config,env,policy):
	if not config['bad_demonstrations']:
		config['obs_data'],action_data,config['r_data'],config['done_data'],index_data,config['targets_data'] =\
			np.load(os.path.join(config['og_path'],f"{config['env_name'][:2]}_demo.npz"),allow_pickle=True).values()
	else:
		config['obs_data'],action_data,config['r_data'],config['done_data'],index_data,config['targets_data'] =\
			np.load(os.path.join(config['og_path'],f"{config['env_name'][:2]}_demo1.npz"),allow_pickle=True).values()

	config['env'] = env.envs[0]
	t_env = window_factory(TransitionFeeder)(config)
	env.envs[0] = t_env
	buffer = IndexReplayBuffer(int(1e6), t_env.observation_space, t_env.action_space, device=policy.device, n_envs=1)

	obs = env.reset()
	action_data,index_data = iter(action_data),iter(index_data)
	for i in range(len(config['obs_data'])):
		action_ep,index_ep = iter(next(action_data,"act_seq_done")),next(index_data,"index_done")		
		done = False
		while not done:
			action = [next(action_ep,"action_done")]
			next_obs,r,done,info = env.step(action)	
			if not done:
				buffer.add(obs,next_obs,action,r,done,index_ep)	
				obs = next_obs
			else:
				prev_obs = obs
				obs = next_obs
				next_obs = [info[0]['terminal_observation']]
				buffer.add(prev_obs,next_obs,action,r,done,index_ep)

	with trange(int(buffer.size()*10/config['batch_size'])) as t:
		for gradient_step in t:
			replay_data = buffer.sample(config['batch_size'], env=env)
			logits,_kwargs = policy.get_action_dist_params(replay_data.observations)
			actor_loss = th.nn.CrossEntropyLoss()(logits,replay_data.indices.squeeze().long())
			with th.no_grad():
				actions = policy(replay_data.observations)
				accuracy = actions.eq(replay_data.indices.squeeze()).float().mean()
			logger.record('pretraining/accuracy',accuracy)
			logger.record('pretraining/log-loss',actor_loss)
			t.set_postfix(accuracy=accuracy.item(),loss=actor_loss.item())

			policy.optimizer.zero_grad()
			actor_loss.backward()
			policy.optimizer.step()

"""Miscellaneous"""
ROLLOUT  = 1
NORM = 2
SIZES2 = 3
SIZES5 = 4
SIZES8 = 5
BLANK = 6
OBS2 = 7
OBS1 = 8
SAMPLE_NAME = {
	ROLLOUT: 'rollout',
	NORM: 'norm',
	SIZES2: 'sizes2',
	SIZES5: 'sizes5',
	SIZES8: 'sizes8',
	BLANK: 'blank',
	OBS2: 'obs2',
	OBS1: 'obs1',
}

class IKAgent:
	def __init__(self,env):
		self.env = env
		self.new_pos = -1
		self.tool_pos = -1
		self.tool = -1
	
	def predict(self,obs):
		joint_states = p.getJointStates(self.env.robot, jointIndices=self.env.robot_left_arm_joint_indices, physicsClientId=self.env.id)
		joint_positions = np.array([x[0] for x in joint_states])

		link_pos = p.getLinkState(self.env.robot, 13, computeForwardKinematics=True, physicsClientId=self.env.id)[0]
		new_pos = p.multiplyTransforms(self.env.target_pos, [0, 0, 0, 1],
					 link_pos-self.env.tool_pos, [0, 0, 0, 1], physicsClientId=self.env.id)[0]

		# new_pos = self.env.target_pos
		new_joint_positions = np.array(p.calculateInverseKinematics(self.env.robot, 13, new_pos, physicsClientId=self.env.id))
		new_joint_positions = new_joint_positions[:7]
		action = 2*(new_joint_positions - joint_positions)

		# p.removeBody(self.new_pos)
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=self.env.id)
		# self.new_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
		# 		basePosition=new_pos, useMaximalCoordinates=False, physicsClientId=self.env.id)

		# p.removeBody(self.tool_pos)
		# sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[10, 255, 10, 1], physicsClientId=self.env.id)
		# self.tool_pos = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=sphere_visual,\
		# 		basePosition=self.env.tool_pos, useMaximalCoordinates=False, physicsClientId=self.env.id)

		return action

class KeyboardAgent:
	key_mappings = {
		'a':'left',
		'd':'right',
		's':'backward',
		'w':'forward',
		'z':'down',
		'x':'up',
	}
	def predict(self,obs):
		key = input('direction: ')
		if key in self.key_mappings:
			return self.key_mappings[key]
		elif key == 'reset':
			return key
		else:
			return None