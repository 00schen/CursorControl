from rl.replay_buffers.modded_buffer import ModdedReplayBuffer
import numpy as np
import warnings

class ModdedTrajReplayBuffer(ModdedReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            block_len,
            env_info_sizes={'episode_success':1},
            # env_info_sizes={'noop':1,},
            sample_base=5000,
    ):
        env_info_sizes.update({'episode_success':1})
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes,
            sample_base=5000*block_len,
        )
        self._max_replay_buffer_size //= block_len
        self.sample_base=sample_base
        self._observations = self._observations.reshape((max_replay_buffer_size//block_len,block_len,-1))
        self._next_obs = self._next_obs.reshape((max_replay_buffer_size//block_len,block_len,-1))
        self._actions = self._actions.reshape((max_replay_buffer_size//block_len,block_len,-1))
        self._rewards = self._rewards.reshape((max_replay_buffer_size//block_len,block_len,-1))
        self._terminals = self._terminals.reshape((max_replay_buffer_size//block_len,block_len,-1))
        for k in self._env_info_keys:
            self._env_infos[k] = self._env_infos[k].reshape((max_replay_buffer_size//block_len,block_len,-1))
        for k in self._obs_dict.keys():
            self._obs_dict[k] = self._obs_dict[k].reshape((max_replay_buffer_size//block_len,block_len,-1))
        for k in self._next_obs_dict.keys():
            self._next_obs_dict[k] = self._next_obs_dict[k].reshape((max_replay_buffer_size//block_len,block_len,-1))

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        for key in self._obs_dict_keys:
            self._obs_dict[key][self._top] = observation[key]
            self._next_obs_dict[key][self._top] = next_observation[key]
        super().add_sample(observation['raw_obs'], action, reward, terminal,
                   next_observation['raw_obs'], env_info=env_info, **kwargs)

    def add_path(self, path):
        self.modify_path(path)
        self._actions[self._top,:len(path['actions'])] = np.array(path['actions'])
        self._rewards[self._top,:len(path['rewards'])] = np.array(path['rewards'])
        self._terminals[self._top,:len(path['terminals'])] = np.array(path['terminals']).reshape((-1,1))
        for i, (
                obs,
                next_obs,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["next_observations"],
            path["env_infos"],
        )):
            self._observations[self._top,i] = obs['raw_obs']
            self._next_obs[self._top,i] = next_obs['raw_obs']
            for key in self._obs_dict_keys:
                self._obs_dict[key][self._top,i] = obs[key]
                self._next_obs_dict[key][self._top,i] = next_obs[key]
            for key in self._env_info_keys:
                self._env_infos[key][self._top,i] = env_info[key]
        
        self._advance()
        self.terminate_episode()

    def _get_batch(self,indices):
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._obs_dict_keys:
            assert key not in batch.keys()
            batch['curr_'+key] = self._obs_dict[key][indices]
            batch['next_'+key] = self._next_obs_dict[key][indices]
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def random_batch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        return self._get_batch(indices)
