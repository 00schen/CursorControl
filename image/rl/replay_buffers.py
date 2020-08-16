from collections import OrderedDict

import numpy as np

from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.envs.env_utils import get_dim

class PavlovReplayBuffer(EnvReplayBuffer):
	def __init__(self,max_replay_buffer_size,env,env_info_sizes=None):
		super().__init__(max_replay_buffer_size,env,env_info_sizes)
		self._inputs = np.zeros((max_replay_buffer_size, 1))

	def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
		self._inputs[self._top] = not kwargs['env_info'].get('noop',True)
		super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)

	def random_batch(self, batch_size):
		indices = np.random.randint(0, self._size, batch_size)
		batch = dict(
			observations=self._observations[indices],
			actions=self._actions[indices],
			rewards=self._rewards[indices],
			terminals=self._terminals[indices],
			next_observations=self._next_obs[indices],
			inputs=self._inputs[indices],
		)
		for key in self._env_info_keys:
			assert key not in batch.keys()
			batch[key] = self._env_infos[key][indices]
		return batch

class BalancedReplayBuffer(PavlovReplayBuffer):
	def __init__(self,max_replay_buffer_size,env,env_info_sizes=None):
		super().__init__(max_replay_buffer_size,env,env_info_sizes)
		self.noop_replay_buffer = PavlovReplayBuffer(max_replay_buffer_size,env,env_info_sizes)

	def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
		if kwargs['env_info'].get('noop',True):
			self.noop_replay_buffer.add_sample(observation, action, reward, terminal, next_observation, **kwargs)
		else:
			super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)

	def random_batch(self, batch_size):
		if self._size >= batch_size//2:
			input_batch = super().random_batch(batch_size//2)
			noop_batch = self.noop_replay_buffer.random_batch(batch_size-batch_size//2)
		else:
			input_batch = super().random_batch(self._size)
			noop_batch = self.noop_replay_buffer.random_batch(batch_size-self._size)
		batch = dict(
			observations=np.concatenate((input_batch['observations'],noop_batch['observations'])),
			actions=np.concatenate((input_batch['actions'],noop_batch['actions'])),
			rewards=np.concatenate((input_batch['rewards'],noop_batch['rewards'])),
			terminals=np.concatenate((input_batch['terminals'],noop_batch['terminals'])),
			next_observations=np.concatenate((input_batch['next_observations'],noop_batch['next_observations'])),
			inputs=np.concatenate((input_batch['inputs'],noop_batch['inputs'])),
		)
		return batch

	def num_steps_can_sample(self):
		return super().num_steps_can_sample() + self.noop_replay_buffer.num_steps_can_sample()

class PavlovSubtrajReplayBuffer(ReplayBuffer):
	def __init__(
		self,
		max_num_traj,
		traj_max,
		subtraj_len,
		env,
	):
		self.env = env
		observation_dim, action_dim = get_dim(env.observation_space), get_dim(env.action_space)
		self._observation_dim = observation_dim
		self._action_dim = action_dim
		self._max_replay_buffer_size = max_num_traj

		self._observations = np.zeros((max_num_traj, traj_max, observation_dim))
		# It's a bit memory inefficient to save the observations twice,
		# but it makes the code *much* easier since you no longer have to
		# worry about termination conditions.
		self._next_obs = np.zeros((max_num_traj, traj_max, observation_dim))
		self._actions = np.zeros((max_num_traj, traj_max, action_dim))
		# Make everything a 2D np array to make it easier for other code to
		# reason about the shape of the data
		self._rewards = np.zeros((max_num_traj, traj_max, 1))
		# self._terminals[i] = a terminal was received at time i
		self._terminals = np.zeros((max_num_traj, traj_max, 1), dtype='uint8')
		self._inputs = np.zeros((max_num_traj, traj_max, 1), dtype='uint8')

		self._sample = 0
		self._index = 0
		self._valid_start_indices = []
		self._subtraj_len = subtraj_len
		self._traj_max = traj_max

	# def add_sample(self, observation, action, reward, next_observation,
	# 			   terminal, env_info, **kwargs):
	# 	self._observations[self._sample][self._index] = observation
	# 	self._actions[self._sample][self._index] = action
	# 	self._rewards[self._sample][self._index] = reward
	# 	self._terminals[self._sample][self._index] = terminal
	# 	self._next_obs[self._sample][self._index] = next_observation

	# 	for key in self._env_info_keys:
	# 		self._env_infos[key][self._sample][self._index] = env_info[key]
	# 	self._advance()

	def terminate_episode(self):
		pass

	def add_sample(self, observation, action, reward, next_observation,
				   terminal, env_info, **kwargs):
		pass

	def add_path(self, path):
		n_items = len(path["observations"])
		if n_items < self._subtraj_len:
			return

		self._observations[self._sample][:n_items] = path['observations']
		self._actions[self._sample][:n_items] = path['actions']
		self._rewards[self._sample][:n_items] = path['rewards']
		self._terminals[self._sample][:n_items] = path['terminals']
		self._next_obs[self._sample][:n_items] = path['next_observations'][...,:]
		self._inputs[self._sample][:n_items] = np.logical_not([env_info['noop'] for env_info in path['env_infos']])[:,np.newaxis]

		self._valid_start_indices.extend([(self._sample,start_index) for start_index in range(n_items-self._subtraj_len+1)])

		self._sample = (self._sample + 1) % self._max_replay_buffer_size

	def random_batch(self, batch_size):
		start_indices = self._valid_start_indices[np.random.choice(
			len(self._valid_start_indices),
			batch_size,
		)]
		batch = dict(
			observations=self.get_subtrajectories(self._observations,start_indices),
			actions=self.get_subtrajectories(self._actions,start_indices),
			rewards=self.get_subtrajectories(self._rewards,start_indices),
			terminals=self.get_subtrajectories(self._terminals,start_indices),
			next_observations=self.get_subtrajectories(self._next_obs,start_indices),
			inputs=self.get_subtrajectories(self._inputs,start_indices),
		)
		return batch

	def get_subtrajectories(self,array,start_indices):
		batch = np.zeros((len(start_indices),self._subtraj_len,array.shape[-1]))
		for i in range(len(batch)):
			index = start_indices[i]
			batch[i] = array[index[0],index[1]:index[1]+self._subtraj_len,:]
		return batch

	def num_steps_can_sample(self):
		return len(self._valid_start_indices)

	def get_diagnostics(self):
		return OrderedDict([
			('size', self._size)
		])

	def __getstate__(self):
		# Do not save self.replay_buffer since it's a duplicate and seems to
		# cause joblib recursion issues.
		return dict(
			observations = self._observations,
			actions = self._actions,
			rewards = self._rewards,
			terminals = self._terminals,
			next_obs = self._next_obs,
			inputs = self._inputs,

			observation_dim = self._observation_dim,
			action_dim = self._action_dim,
			max_replay_buffer_size = self._max_replay_buffer_size,
			sample=self._sample,
			index=self._index,
			valid_start_indices=self._valid_start_indices,
			subtraj_len=self._subtraj_len,
			traj_max=self._traj_max,
		)

	def __setstate__(self, d):
		self._observation_dim = d["observation_dim"]
		self._action_dim = d["action_dim"]
		self._max_replay_buffer_size = d["max_replay_buffer_size"]
		
		self._observations = d["observations"]
		self._next_obs = d["next_obs"]
		self._actions = d["actions"]
		self._rewards = d["rewards"]
		self._terminals = d["terminals"]
		self._inputs = d['inputs']

		self._sample = d['sample']
		self._index = d['index']
		self._valid_start_indices = d['valid_start_indices']
		self._subtraj_len = d['subtraj_len']
		self._traj_max = d['traj_max']
