import numpy as np
import h5py
import os
from pathlib import Path
from .balanced_buffer import BalancedReplayBuffer

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

	# def add_path(self, path):
	# 	target_index = path['env_infos'][0]['unique_index']
	# 	self.key_index_limits[int(target_index)] += 1
	# 	if True in [info['target1_reached'] for info in path['env_infos']]:
	# 		i = min([i for i in range(len(path['env_infos'])) if path['env_infos'][i]['target1_reached']])
	# 		target_index = path['env_infos'][i]['unique_index']
	# 		self.key_index_limits[int(target_index)] += 1
	# 	return super().add_path(path)

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
		if 'curr_gaze_features' in batch:
			batch['curr_gaze_features'] = gaze_samples
			batch['next_gaze_features'] = gaze_samples
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

