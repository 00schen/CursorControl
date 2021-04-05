from gym.spaces import Discrete

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np
import warnings
import h5py
import os
from pathlib import Path

class BalancedReplayBuffer(EnvReplayBuffer):
	def __init__(
			self,
			max_replay_buffer_size,
			env,
			target_name='noop',
			# env_info_sizes={'noop':1,'episode_success':1},
			env_info_sizes={'noop':1,},
			false_prop=.5,
	):
		env_info_sizes.update({'noop':1,})
		# env_info_sizes.update({'noop':1,'episode_success':1})
		super().__init__(
			max_replay_buffer_size=max_replay_buffer_size,
			env=env,
			env_info_sizes=env_info_sizes
		)
		self.target_name = target_name
		self.false_prop = false_prop

	# def modify_path(self,path):
		# for info in path['env_infos']:
		# 	# info['episode_success'] = path['env_infos'][-1]['task_success']
		# 	info['episode_success'] = len(path['env_infos']) < 150

	def terminate_episode(self):
		if self.target_name in self._env_infos:
			record = self._env_infos[self.target_name]
		elif self.target_name == 'terminals':
			record = self._terminals
		self.true_indices = np.arange(self._size)[record[:self._size].flatten().astype(bool)]
		self.false_indices = np.arange(self._size)[record[:self._size].flatten().astype(bool) != True]

	def random_batch(self, batch_size):
		true_batch_size = batch_size - int(self.false_prop*batch_size)
		true_indices = self.env.rng.choice(self.true_indices, size=true_batch_size, replace=self._replace or self.true_indices < true_batch_size)
		false_batch_size = int(self.false_prop*batch_size)
		false_indices = self.env.rng.choice(self.false_indices, size=false_batch_size, replace=self._replace or self.false_indices < false_batch_size)
		indices = np.concatenate((true_indices,false_indices))
		if not self._replace and self._size < batch_size:
			warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
		batch = dict(
			observations=self._observations[indices],
			actions=self._actions[indices],
			rewards=self._rewards[indices],
			terminals=self._terminals[indices],
			next_observations=self._next_obs[indices],
		)
		for key in self._env_info_keys:
			assert key not in batch.keys()
			batch[key] = self._env_infos[key][indices]
		return batch

class GazeReplayBuffer(BalancedReplayBuffer):
	def __init__(
			self,
			data_path,
			*args,
			**kwargs
	):
		kwargs['env_info_sizes'].update({'unique_index': 1})
		super().__init__(
			*args,
			**kwargs
		)
		with h5py.File(os.path.join(str(Path(__file__).resolve().parents[2]),'gaze_capture','gaze_data',data_path),'r') as gaze_data:
			self.gaze_dataset = {k:v[()] for k,v in gaze_data.items()}
		self.key_index_limits = [0]*len(self.gaze_dataset.keys())

	def add_path(self, path):
		target_index = path['env_infos'][0]['unique_index']
		self.key_index_limits[int(target_index)] += 1
		if True in [info['target1_reached'] for info in path['env_infos']]:
			i = min([i for i in range(len(path['env_infos'])) if path['env_infos'][i]['target1_reached']])
			target_index = path['env_infos'][i]['unique_index']
			self.key_index_limits[int(target_index)] += 1
		return super().add_path(path)

	def sample_gaze(self,indices):
		samples = []
		for index in indices:
			index = int(index)
			data = self.gaze_dataset[str(index)]
			data_ind = np.random.choice(len(data))
			# data_ind = np.random.choice(min(len(data),self.key_index_limits[index]))
			samples.append(data[data_ind].flatten())
		return samples

	def random_batch(self, batch_size):
		batch = super().random_batch(batch_size)
		gaze_samples = np.array(self.sample_gaze(batch['unique_index'].flatten()))
		batch['observations'][:,-128:] = gaze_samples
		batch['next_observations'][:,-128:] = gaze_samples
		return batch

class ContGazeReplayBuffer(BalancedReplayBuffer):
	def __init__(
			self,
			data_path,
			*args,
			**kwargs
	):
		kwargs['env_info_sizes'].update({'target1_reached': 1, 'tool_pos': 3})
		super().__init__(
			*args,
			**kwargs
		)
		with h5py.File(os.path.join(str(Path(__file__).resolve().parents[2]),'gaze_capture','gaze_data','bottle_cont_gaze_data.h5'),'r') as gaze_data:
			self.gaze_dataset = {k:v[()] for k,v in gaze_data.items()}
			self.gaze_dataset['sub1_gaze'] = self.gaze_dataset['gaze_features'][self.gaze_dataset['target1_reached']]
			self.gaze_dataset['sub2_gaze'] = self.gaze_dataset['gaze_features'][np.logical_not(self.gaze_dataset['target1_reached'])]
			self.gaze_dataset['sub1_tool'] = self.gaze_dataset['tool_pos'][self.gaze_dataset['target1_reached']]
			self.gaze_dataset['sub2_tool'] = self.gaze_dataset['tool_pos'][np.logical_not(self.gaze_dataset['target1_reached'])]
		self.key_index_limits = [0]*len(self.gaze_dataset.keys())

	def sample_gaze(self,indices):
		indices = indices.astype(bool)
		gazes = np.zeros((len(indices),128))
		tools = np.zeros((len(indices),3))
		sub1_indices = np.arange(len(indices))[np.logical_not(indices)]
		sub2_indices = np.arange(len(indices))[indices]
		data_ind = np.random.choice(len(sub1_indices))
		data = self.gaze_dataset[f"sub1_gaze"]
		gazes[sub1_indices] = data[data_ind]
		data = self.gaze_dataset[f"sub1_tool"]
		tools[sub1_indices] = data[data_ind]
		data_ind = np.random.choice(len(sub2_indices))
		data = self.gaze_dataset[f"sub2_gaze"]
		gazes[sub2_indices] = data[data_ind]
		data = self.gaze_dataset[f"sub2_tool"]
		tools[sub2_indices] = data[data_ind]
		return gazes,tools

	def random_batch(self, batch_size):
		batch = super().random_batch(batch_size)
		gaze_samples, tool_pos = self.sample_gaze(batch['target1_reached'].flatten())
		batch['observations'][:,-128:] = gaze_samples
		batch['next_observations'][:,-128:] = gaze_samples

		batch['rewards'] = np.exp(-50*np.linalg.norm(tool_pos-batch['tool_pos']))*(1+batch['target1_reached'].flatten())/2
		batch['rewards'] = batch['rewards'][:,None]

		return batch

