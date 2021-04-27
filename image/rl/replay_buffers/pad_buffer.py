import numpy as np
import warnings

def pad_buffer_factory(base):
	class PadReplayBuffer(base):
		def __init__(
				self,
				full_size,
				*args,
				**kwargs,
		):
			super().__init__(
				*args,**kwargs
			)
			self.full_size = full_size

		def random_batch(self, batch_size):
			batch = super().random_batch(batch_size)
			if 'curr_goal' in batch:
				batch_shape = batch['curr_goal'].shape
				batch['curr_goal'] = np.concatenate((batch['curr_goal'],
										np.zeros((*batch_shape[:-1],self.full_size-batch_shape[-1],))),axis=-1)
				batch['next_goal'] = np.concatenate((batch['next_goal'],
										np.zeros((*batch_shape[:-1],self.full_size-batch_shape[-1],))),axis=-1)
			return batch
	
	return PadReplayBuffer
