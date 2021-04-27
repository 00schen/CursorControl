from collections import OrderedDict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import nn

class DQNTrainer(TorchTrainer):
	def __init__(
			self,
			qf,
			target_qf,
			optimizer,
			learning_rate=1e-3,
			soft_target_tau=1e-3,
			target_update_period=1,
			qf_criterion=None,

			discount=0.99,
	):
		super().__init__()
		self.qf = qf
		self.target_qf = target_qf
		self.learning_rate = learning_rate
		self.soft_target_tau = soft_target_tau
		self.target_update_period = target_update_period
		self.qf_optimizer = optimizer
		self.discount = discount
		self.qf_criterion = qf_criterion or nn.MSELoss()
		self.eval_statistics = OrderedDict()
		self._n_train_steps_total = 0
		self._need_to_update_eval_statistics = True

	def get_diagnostics(self):
		return self.eval_statistics

	def end_epoch(self, epoch):
		self._need_to_update_eval_statistics = True