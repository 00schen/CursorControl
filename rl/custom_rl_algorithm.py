from rlkit.torch import TorchBatchRLAlgorithm

class CustomBatchRLAlgorithm(TorchBatchRLAlgorithm):
	def __init__(self, *args, **kwargs):
		self.pretrain = kwargs.pop('pretrain',False)
		self.num_pretrains = kwargs.pop('num_pretrains',0)
		super().__init__(*args,**kwargs)

	def _train(self):
		if self.pretrain:
			for _ in range(self.num_pretrains):
				train_data = self.replay_buffer.random_batch(self.batch_size)
				self.trainer.train(train_data)
		super()._train()
