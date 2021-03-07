from gym.spaces import Discrete

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np
import warnings

class BalancedReplayBuffer(EnvReplayBuffer):
	def __init__(
			self,
			max_replay_buffer_size,
			env,
			target_name='noop',
			env_info_sizes={'noop':1,'episode_success':1},
			false_prop=.5,
	):
		env_info_sizes.update({'noop':1,'episode_success':1})
		super().__init__(
			max_replay_buffer_size=max_replay_buffer_size,
			env=env,
			env_info_sizes=env_info_sizes
		)
		self.target_name = target_name
		self.false_prop = false_prop

	def modify_path(self,path):
		for info in path['env_infos']:
			# info['episode_success'] = path['env_infos'][-1]['task_success']
			info['episode_success'] = len(path['env_infos']) < 150

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
