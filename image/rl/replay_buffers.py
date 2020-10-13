from railrl.data_management.simple_replay_buffer import SimpleReplayBuffer
from railrl.envs.env_utils import get_dim

class PavlovReplayBuffer(SimpleReplayBuffer):
	def __init__(self,max_replay_buffer_size,env):
		self.env = env
		super().__init__(max_replay_buffer_size,get_dim(env.observation_space),get_dim(env.action_space),{})
		
	def add_sample(self, observation, action, reward, next_observation, terminal, env_info, **kwargs):
		self._observations[self._top] = observation
		self._actions[self._top] = action
		self._rewards[self._top] = reward
		self._terminals[self._top] = env_info['task_success']
		self._next_obs[self._top] = next_observation

		for key in self._env_info_keys:
			self._env_infos[key][self._top] = env_info[key]
		self._advance()
