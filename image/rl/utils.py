import numpy as np
import torch as th
from replay_buffers import PavlovReplayBuffer

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
class AdaptPathLoader(DictToMDPPathLoader):
	def load_path(self, path, replay_buffer, obs_dict=None):
		replay_buffer.add_path(path,True)
		# super().load_path(path, replay_buffer, obs_dict)

class AdaptReplayBuffer(PavlovReplayBuffer):
	def __init__(self,max_replay_buffer_size,env):
		super().__init__(max_replay_buffer_size,env)
	def add_path(self,path,adapt_tran=False):
		self.adapt_path(path,adapt_tran)
		super().add_path(path)
	def adapt_path(self,path,adapt_tran):
		tran_iter = zip(list(path['observations'][1:])+[path['next_observations'][-1]],
						list(path['actions']),
						list(path['rewards']),
						list(path['terminals']),
						list(path['env_infos']),
						list(path['agent_infos']),
					)

		processed_trans = []
		obs = path['observations'][0]
		if adapt_tran:
			obs = self.env.adapt_reset(obs)
		for next_obs,action,r,done,info,ainfo in tran_iter:
			if adapt_tran:
				next_obs,r,done,info = self.env.adapt_step(next_obs,r,done,info)
			action = ainfo.get(self.env.action_type,action)
			processed_trans.append((obs,next_obs,action,r,done,info,ainfo))
			obs = next_obs

		new_path = dict(zip(
			['observations','next_observations','actions','rewards','terminals','env_infos','agent_infos'],
			list(zip(*processed_trans))
			))
		path.update(new_path)
		path['observations'] = np.array(path['observations'])
		path['next_observations'] = np.array(path['next_observations'])
		path['actions'] = np.array(path['actions'])
		path['rewards'] = np.array(path['rewards'])[:,np.newaxis]
		path['terminals'] = np.array(path['terminals'])[:,np.newaxis]
		return path

class ThRunningMeanStd:
	def __init__(self, epsilon=1, shape=()):
		"""
		Calulates the running mean and std of a data stream
		https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
		:param epsilon: helps with arithmetic issues
		:param shape: the shape of the data stream's output
		"""
		self.mean = th.zeros(shape)
		self.var = th.ones(shape)
		self.count = epsilon

	def update(self, arr):
		batch_mean = th.mean(arr, dim=0)
		batch_var = th.var(arr, dim=0)
		batch_count = arr.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		delta = batch_mean - self.mean
		tot_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / tot_count
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		m_2 = m_a + m_b + th.square(delta) * self.count * batch_count / (self.count + batch_count)
		new_var = m_2 / (self.count + batch_count)

		new_count = batch_count + self.count

		self.mean = new_mean
		self.var = new_var
		self.count = new_count

class RunningMeanStd:
	def __init__(self, epsilon=1, shape=()):
		"""
		Calulates the running mean and std of a data stream
		https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
		:param epsilon: helps with arithmetic issues
		:param shape: the shape of the data stream's output
		"""
		self.mean = np.zeros(shape)
		self.var = np.ones(shape)
		self.count = epsilon

	def update(self, arr):
		batch_mean = np.mean(arr, axis=0)
		batch_var = np.var(arr, axis=0)
		batch_count = arr.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		delta = batch_mean - self.mean
		tot_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / tot_count
		m_a = self.var * self.count
		m_b = batch_var * batch_count
		m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
		new_var = m_2 / (self.count + batch_count)

		new_count = batch_count + self.count

		self.mean = new_mean
		self.var = new_var
		self.count = new_count
