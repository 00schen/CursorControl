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
            obs_feature_sizes={},
            sample_base=0,
            latent_size=3,
            store_latents=True,
            window_size=None,
    ):
        env_info_sizes.update({'episode_success': 1})
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            env=env,
            env_info_sizes=env_info_sizes
        )
        self._obs_dict = {}
        self._next_obs_dict = {}
        self._obs_dict_keys = set(env.feature_sizes.keys()) | set(['goal'])
        self.feature_keys = env.feature_sizes.keys()

        iter_dict = {'goal': env.goal_size}
        iter_dict.update(env.feature_sizes)
        iter_dict.update(obs_feature_sizes)

        # window functionality only works if replay buffer never becomes full / old entries never discarded
        self.window_size = window_size
        if self.window_size is not None:
            self._hist_indices = []
            # for key in env.feature_sizes.keys():
            #     iter_dict[key] *= window
            # self._obs_dict_keys.add('obs_hist')
            # iter_dict['obs_hist'] = self._observation_dim * window
            # self._obs_dict_keys.add('hist_mask')
            # iter_dict['hist_mask'] = window

        if store_latents:
            self._obs_dict_keys.add('latents')
            iter_dict['latents'] = latent_size

        for key, size in iter_dict.items():
            self._obs_dict[key] = np.zeros((max_replay_buffer_size, size))
            self._next_obs_dict[key] = np.zeros((max_replay_buffer_size, size))

        # for envs with goal sets separate from observation
        if hasattr(env.base_env, 'goal_set_shape'):
            self._obs_dict_keys.add('goal_set')
            self._obs_dict['goal_set'] = np.zeros((max_replay_buffer_size,) + env.base_env.goal_set_shape)
            self._next_obs_dict['goal_set'] = np.zeros((max_replay_buffer_size,) + env.base_env.goal_set_shape)

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
        for key in self._obs_dict_keys:
            if key == 'latents':
                self._obs_dict[key][self._top] = observation[key][:4]      
            self._obs_dict[key][self._top] = observation[key]
            if key in next_observation.keys():
                self._next_obs_dict[key][self._top] = next_observation[key]
            else:
                self._next_obs_dict[key][self._top] = None
        super().add_sample(observation['raw_obs'], action, reward, terminal,
                           next_observation['raw_obs'], env_info=env_info, **kwargs)

    def _get_batch(self, batch_size):
        if self._size < batch_size:
            batch_size = self._size
        indices = np.random.choice(self._size, size=batch_size, replace=self._replace)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._obs_dict_keys:
            assert key not in batch.keys()
            batch['curr_' + key] = self._obs_dict[key][indices]
            batch['next_' + key] = self._next_obs_dict[key][indices]
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        if self.window_size is not None:
            hist_indices = [self._hist_indices[idx] for idx in indices]
            obs_hist = [self._observations[hist_idx] for hist_idx in hist_indices]
            batch['obs_hist'] = np.stack([np.pad(x, ((self.window_size - len(x), 0), (0, 0))) for x in obs_hist],
                                         axis=0)
            for key in self.feature_keys:
                key_hist = [self._obs_dict[key][hist_idx] for hist_idx in hist_indices]
                batch[key + '_hist'] = np.stack([np.pad(x, ((self.window_size - len(x), 0), (0, 0)))
                                                 for x in key_hist], axis=0)

            batch['hist_mask'] = np.stack([np.concatenate((np.zeros(self.window_size - len(hist_idx)),
                                                           np.ones(len(hist_idx)))) for hist_idx in hist_indices],
                                          axis=0)

        return batch

    def random_batch(self, batch_size):
        # if self._size < batch_size:
        #     batch_size = self._size
        # indices = np.random.choice(self._size, size=batch_size, replace=self._replace)

        # indices = np.random.choice(self._size, size=batch_size, replace=self._replace or self._size < batch_size)
        # if not self._replace and self._size < batch_size:
        #     warnings.warn(
        #         'Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
        return self._get_batch(batch_size)

    def add_path(self, path):
        self.modify_path(path)
        hist_indices = [self._top]
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )

            if self.window_size is not None:
                self._hist_indices.append(hist_indices.copy())
                hist_indices.append(self._top)
                hist_indices = hist_indices[-self.window_size:]
        self.terminate_episode()
