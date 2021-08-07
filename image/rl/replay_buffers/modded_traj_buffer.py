from rl.replay_buffers.modded_buffer import ModdedReplayBuffer
import numpy as np
import warnings


class ModdedTrajReplayBuffer(ModdedReplayBuffer):
    def _get_batch(self, batch_size=None):
        # first choose some indices
        indices = np.random.choice(self._size, size=10, replace=self._replace or self._size < 10)
        sub_goals = self._obs_dict['sub_goal'][:self._size]
        unique_targets = np.unique(sub_goals[indices], axis=0)
        # get some samples with close targets
        indices = []
        for target in unique_targets:
            target_indices = np.argwhere(np.linalg.norm(target-sub_goals, axis=1) < .02).squeeze()
            target_indices = np.random.choice(target_indices, size=50, replace=True)
            indices.append(target_indices)
        # get the full samples
        batch = dict(
            observations=np.array([self._observations[ind] for ind in indices]),
            actions=np.array([self._actions[ind] for ind in indices]),
            rewards=np.array([self._rewards[ind] for ind in indices]),
            terminals=np.array([self._terminals[ind] for ind in indices]),
            next_observations=np.array([self._next_obs[ind] for ind in indices]),
        )
        for key in self._obs_dict_keys:
            assert key not in batch.keys()
            batch['curr_' + key] = np.array([self._obs_dict[key][ind] for ind in indices])
            batch['next_' + key] = np.array([self._next_obs_dict[key][ind] for ind in indices])
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = np.array([self._env_infos[key][ind] for ind in indices])
        return batch
