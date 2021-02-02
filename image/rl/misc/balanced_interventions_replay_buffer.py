from gym.spaces import Discrete

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np

class BalancedSuccessReplayBuffer(EnvReplayBuffer):
	def __init__(
			self,
			max_replay_buffer_size,
			env,
			env_info_sizes={'noop':1,'episode_success':1},
			fail_prop=.5,
	):
		env_info_sizes.update({'noop':1,'episode_success':1})
		super().__init__(
			max_replay_buffer_size=max_replay_buffer_size,
			env=env,
			env_info_sizes=env_info_sizes
		)
		self.fail_prop = fail_prop

	def modify_path(self,path):
		for info in path['env_infos']:
			info['episode_success'] = path['env_infos'][-1]['task_success']

	def terminate_episode(self):
		self.success_indices = np.arange(self._size)[self._env_infos['episode_success'][:self._size].flatten().astype(bool)]
		self.fail_indices = np.arange(self._size)[self._env_infos['episode_success'][:self._size].flatten().astype(bool) != True]

	def random_batch(self, batch_size):
		success_batch_size = batch_size - int(self.fail_prop*batch_size)
		success_indices = np.random.choice(self.success_indices, size=success_batch_size, replace=self._replace or self.success_indices < success_batch_size)
		fail_batch_size = int(self.fail_prop*batch_size)
		fail_indices = np.random.choice(self.fail_indices, size=fail_batch_size, replace=self._replace or self.fail_indices < fail_batch_size)
		indices = np.concatenate((success_indices,fail_indices))
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

class BalancedInterReplayBuffer(EnvReplayBuffer):
	def __init__(
			self,
			max_replay_buffer_size,
			env,
			env_info_sizes={'noop':1,'episode_success':1},
			inter_prop=.5,
	):
		env_info_sizes.update({'noop':1,'episode_success':1})
		super().__init__(
			max_replay_buffer_size=max_replay_buffer_size,
			env=env,
			env_info_sizes=env_info_sizes
		)
		self.inter_prop = inter_prop

	def modify_path(self,path):
		for info in path['env_infos']:
			info['episode_success'] = path['env_infos'][-1]['task_success']

	def terminate_episode(self):
		self.noop_indices = np.arange(self._size)[self._env_infos['noop'][:self._size].flatten().astype(bool)]
		self.inter_indices = np.arange(self._size)[self._env_infos['noop'][:self._size].flatten().astype(bool) != True]

	def random_batch(self, batch_size):
		noop_batch_size = batch_size - int(self.inter_prop*batch_size)
		noop_indices = np.random.choice(self.noop_indices, size=noop_batch_size, replace=self._replace or self.noop_indices < noop_batch_size)
		inter_batch_size = int(self.inter_prop*batch_size)
		inter_indices = np.random.choice(self.inter_indices, size=inter_batch_size, replace=self._replace or self.inter_indices < inter_batch_size)
		indices = np.concatenate((noop_indices,inter_indices))
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
