from rlkit.samplers.data_collector import MdpPathCollector
import numpy as np

class FullPathCollector(MdpPathCollector):
	def collect_new_paths(
		self,
		max_path_length,
		num_steps,
		discard_incomplete_paths=False,
	):
		paths = []
		num_steps_collected = 0
		while num_steps_collected < num_steps:
			path = self._rollout_fn(
				self._env,
				self._policy,
				max_path_length=max_path_length,
				render=self._render,
				render_kwargs=self._render_kwargs,
			)
			path_len = len(path['actions'])
			num_steps_collected += path_len
			# path['terminals'] = np.logical_or(path['rewards'] < 0, path['terminals'])
			paths.append(path)
		self._num_paths_total += len(paths)
		self._num_steps_total += num_steps_collected
		self._epoch_paths.extend(paths)
		return paths
		