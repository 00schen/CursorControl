from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import numpy as np


class BalancedReplayBuffer(EnvReplayBuffer):
    def random_balanced_batch(self, batch_size):
        half_size = batch_size // 2
        pos_idx = np.where(self._rewards.flatten() >= 0)[0]
        n_pos = len(pos_idx)
        pos_rand_idx = pos_idx[np.random.choice(n_pos, size=half_size, replace=self._replace or n_pos < half_size)]
        neg_idx = np.where(self._rewards.flatten() < 0)[0]
        n_neg = len(neg_idx)
        neg_rand_idx = neg_idx[np.random.choice(n_neg, size=half_size, replace=self._replace or n_neg < half_size)]
        idx = np.concatenate((pos_rand_idx, neg_rand_idx))
        batch = dict(
            observations=self._observations[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            terminals=self._terminals[idx],
            next_observations=self._next_obs[idx],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][idx]
        return batch
