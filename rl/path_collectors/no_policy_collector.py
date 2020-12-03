from collections import deque, OrderedDict
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector import PathCollector

class NoPolicyPathCollector(PathCollector):
    def __init__(
        self,
        env,
        max_num_epoch_paths_saved=None,
        render=False,
        render_kwargs=None,
        rollout_fn=rollout,
        save_env_in_snapshot=True,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn

        self._num_steps_total = 0
        self._num_paths_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot

    def collect_new_paths(
            self, policy_fn, max_path_length, 
            num_steps, discard_incomplete_paths
        ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            path = self._rollout_fn(
                self._env,
                policy_fn,
                max_path_length=max_path_length,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            path_len = len(path['actions'])
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths
    
    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        snapshot_dict = dict()
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict
