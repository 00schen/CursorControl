from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np
import warnings
import h5py
import os
from pathlib import Path
from .modded_buffer import ModdedReplayBuffer

class HERReplayBuffer(ModdedReplayBuffer):
	def __init__(
			self,
			max_replay_buffer_size,
			env,
			# env_info_sizes={'episode_success':1, 'target1_reached': 1},
			env_info_sizes={'episode_success':1, },
			sample_base=5000*200,
	):
		# env_info_sizes.update({'episode_success':1, 'target1_reached': 1})
		env_info_sizes.update({'episode_success':1,})
		super().__init__(
			max_replay_buffer_size=max_replay_buffer_size,
			env=env,
			env_info_sizes=env_info_sizes,
			sample_base=sample_base
		)
		self._hindsight_goals = np.zeros((max_replay_buffer_size, 16, env.feature_sizes['goal']))
		self._hindsight_success = np.zeros((max_replay_buffer_size, 16, 1))

	def add_sample(self, observation, action, reward, next_observation,
				   terminal, env_info, **kwargs):
		self._hindsight_goals[self._top] = env_info['hindsight_goals']
		self._hindsight_success[self._top] = env_info['hindsight_success']
		super().add_sample(observation, action, reward, next_observation,
				   terminal, env_info=env_info, **kwargs)

	def add_path(self, path):
		# path['terminals'][-1] = path['env_infos'][-1]['target1_reached']
		# path['rewards'][-1] = -1 + .5*(path['env_infos'][-1]['target1_reached']+1)
		for i,(
			info,
			obs,
			next_obs,
		) in enumerate(zip(
			path['env_infos'],
			path["observations"],
			path["next_observations"],
		)):
			hindsight_goals = np.random.randint(low=i,high=len(path['rewards']),size=16)
			info['hindsight_success'] = np.array([ind==i for ind in hindsight_goals])[:,None]
			info['hindsight_goals'] = np.array([path["next_observations"][ind]['hindsight_goal'] for ind in hindsight_goals])
		super().add_path(path)

	def _get_batch(self,indices):
		batch = super()._get_batch(indices)
		for k,v in batch.items():
			batch[k] = np.tile(v,(2,1))
		hindsight_ind = np.random.randint(low=0,high=16,size=len(indices))
		batch['curr_goal'][:len(indices)] = self._hindsight_goals[indices,hindsight_ind]
		batch['next_goal'][:len(indices)] = self._hindsight_goals[indices,hindsight_ind]
		hindsight_success = self._hindsight_success[indices,hindsight_ind]
		# batch['terminals'][:len(indices)] = np.where(hindsight_success,batch['target1_reached'][:len(indices)],batch['terminals'][:len(indices)])
		# batch['rewards'][:len(indices)] = np.where(hindsight_success,-1 + .5*(batch['target1_reached'][:len(indices)]+1),batch['rewards'][:len(indices)])
		batch['terminals'][:len(indices)] = np.where(hindsight_success,True,batch['terminals'][:len(indices)])
		batch['rewards'][:len(indices)] = np.where(hindsight_success,0,batch['rewards'][:len(indices)])

		return batch

