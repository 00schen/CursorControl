from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import numpy as np
import warnings


class ModdedReplayBuffer(EnvReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes={'episode_success': 1},
            # env_info_sizes={'noop':1,},
            sample_base=0,
            latent_size=3,
            store_latents=True
    ):
        env_info_sizes.update({'episode_success': 1})
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes
        )
        self._obs_dict = {}
        self._next_obs_dict = {}
        self._obs_dict_sizes = {
            'ground_truth': env.base_goal_size,
            'goal_obs': env.goal_space.low.size,
        }

        if store_latents:
            self._obs_dict_keys.add('latents')
            iter_dict['latents'] = latent_size

        for key, size in self._obs_dict_sizes.items():
            self._obs_dict[key] = np.zeros((max_replay_buffer_size, size))
            self._next_obs_dict[key] = np.zeros((max_replay_buffer_size, size))

        self.sample_base = sample_base

    def _advance(self):
        self._top = ((self._top + 1 - self.sample_base) % (
                    self._max_replay_buffer_size - self.sample_base)) + self.sample_base \
            if self._top > self.sample_base else self._top + 1
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def modify_path(self, path):
        for info in path['env_infos']:
            info['episode_success'] = path['env_infos'][-1]['task_success']

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        for key in self._obs_dict_sizes.keys():
            self._obs_dict[key][self._top] = np.array([observation[key]]).ravel()
            if key in next_observation.keys():
                self._next_obs_dict[key][self._top] = np.array([next_observation[key]]).ravel()
            else:
                self._next_obs_dict[key][self._top] = None
        super().add_sample(observation['raw_obs'], action, reward, terminal,
                           next_observation['raw_obs'], env_info=env_info, **kwargs)

    def _get_batch(self, indices):
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._obs_dict_sizes.keys():
            assert key not in batch.keys()
            batch['curr_' + key] = self._obs_dict[key][indices]
            batch['next_' + key] = self._next_obs_dict[key][indices]
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def random_batch(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            warnings.warn(
                'Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        return self._get_batch(indices)
