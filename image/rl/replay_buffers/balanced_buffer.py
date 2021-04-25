import numpy as np
import warnings

def balanced_buffer_factory(base):
	class BalancedReplayBuffer(base):
		def __init__(
				self,
				*args,
				target_name='noop',
				false_prop=.5,
				**kwargs,
		):
			super().__init__(
				*args,**kwargs
			)
			self.target_name = target_name
			self.false_prop = false_prop

		def terminate_episode(self):
			if self.target_name in self._env_infos:
				record = self._env_infos[self.target_name]
			elif self.target_name == 'terminals':
				record = self._terminals
			self.true_indices = np.arange(self._size)[record[:self._size].flatten().astype(bool)]
			self.false_indices = np.arange(self._size)[record[:self._size].flatten().astype(bool) != True]

		def random_batch(self, batch_size):
			true_batch_size = batch_size - int(self.false_prop*batch_size)
			true_indices = self.env.rng.choice(self.true_indices, size=true_batch_size, replace=self._replace or self.true_indices < true_batch_size)
			false_batch_size = int(self.false_prop*batch_size)
			false_indices = self.env.rng.choice(self.false_indices, size=false_batch_size, replace=self._replace or self.false_indices < false_batch_size)
			
			indices = np.concatenate((true_indices,false_indices))
			if not self._replace and self._size < batch_size:
				warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
			return self._get_batch(indices)
	
	return BalancedReplayBuffer

def balanced_traj_buffer_factory(base):
	class BalancedReplayBuffer(base):
		def __init__(
				self,
				*args,
				target_name='noop',
				false_prop=.5,
				**kwargs,
		):
			super().__init__(
				*args,**kwargs
			)
			self.target_name = target_name
			self.false_prop = false_prop

		def terminate_episode(self):
			if self.target_name in self._env_infos:
				record = self._env_infos[self.target_name]
			elif self.target_name == 'terminals':
				record = self._terminals
			self.true_indices = np.arange(self._size)[record[:self._size,-1].flatten().astype(bool)]
			self.false_indices = np.arange(self._size)[record[:self._size,-1].flatten().astype(bool) != True]

		def random_batch(self, batch_size):
			true_batch_size = batch_size - int(self.false_prop*batch_size)
			true_indices = self.env.rng.choice(self.true_indices, size=true_batch_size, replace=self._replace or self.true_indices < true_batch_size)
			false_batch_size = int(self.false_prop*batch_size)
			false_indices = self.env.rng.choice(self.false_indices, size=false_batch_size, replace=self._replace or self.false_indices < false_batch_size)
			
			indices = np.concatenate((true_indices,false_indices))
			if not self._replace and self._size < batch_size:
				warnings.warn('Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.')
			return self._get_batch(indices)
	
	return BalancedReplayBuffer
