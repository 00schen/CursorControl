import numpy as np
import torch as th
import torch.optim as optim
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core import logger
from .mixup import mixup
from collections import OrderedDict

class RewardTrainer(TorchTrainer):
	def __init__(
			self,
			rf,
			target_name='reward',
			rf_lr=1e-3,
			optimizer_class=optim.Adam,
			use_mixup=False,
	):
		super().__init__()
		self.rf = rf
		self.target_name = target_name
		self.rf_criterion = F.binary_cross_entropy_with_logits
		self.rf_optimizer = optimizer_class(
			self.rf.parameters(),
			lr=rf_lr,
		)
		self.eval_statistics = OrderedDict()
		self._n_train_steps_total = 0
		self._need_to_update_eval_statistics = True
		self.use_mixup = use_mixup

	def train_from_torch(self, batch):
		rewards = batch[self.target_name]
		obs = batch['observations']
		next_obs = batch['next_observations']

		"""
		Reward and R loss
		"""
		if self.use_mixup:
			(train_obs,train_next_obs),train_rewards = mixup([obs,next_obs],rewards)
		else:
			train_obs,train_next_obs,train_rewards = obs,next_obs,rewards
		rf_pred = self.rf(train_obs,train_next_obs)
		rf_loss = self.rf_criterion(rf_pred, train_rewards)

		"""
		Update Q networks
		"""
		self.rf_optimizer.zero_grad()
		rf_loss.backward()
		self.rf_optimizer.step()

		"""
		Save some statistics for eval using just one batch.
		"""
		stat_pred = self.rf(obs,next_obs).detach()
		accuracy = th.eq(stat_pred,rewards).float().mean()

		if self._need_to_update_eval_statistics:
			self._need_to_update_eval_statistics = False
			self.eval_statistics['RF Loss'] = np.mean(ptu.get_numpy(rf_loss))
			self.eval_statistics['RF Accuracy'] = np.mean(ptu.get_numpy(accuracy))

	def get_diagnostics(self):
		return self.eval_statistics

	def end_epoch(self, epoch):
		self._need_to_update_eval_statistics = True

	@property
	def networks(self):
		nets = [
			self.rf,
		]
		return nets

	def get_snapshot(self):
		return dict(
			rf = self.rf,
		)
