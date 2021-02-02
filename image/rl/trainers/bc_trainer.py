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

class DisBCTrainer(TorchTrainer):
	def __init__(
			self,
			policy,
			policy_lr=1e-3,
			optimizer_class=optim.Adam,
			use_mixup=False,
	):
		super().__init__()
		self.policy = policy
		self.bc_criterion = lambda pred,target: -(target*F.log_softmax(pred,dim=1)).sum(dim=1).mean()
		self.policy_optimizer = optimizer_class(
			self.policy.parameters(),
			lr=policy_lr,
		)
		self.eval_statistics = OrderedDict()
		self._n_train_steps_total = 0
		self._need_to_update_eval_statistics = True
		self.use_mixup = use_mixup

	def train_from_torch(self, batch):
		noop = th.clamp(batch['rewards']+1,0,1)
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		"""
		Reward and R loss
		"""
		if self.use_mixup:
			(train_obs,),train_actions = mixup([obs],actions)
		else:
			train_obs,train_actions = obs,actions
		bc_pred = self.policy(train_obs)
		bc_loss = self.bc_criterion(bc_pred, train_actions)

		"""
		Update Q networks
		"""
		self.policy_optimizer.zero_grad()
		bc_loss.backward()
		self.policy_optimizer.step()

		"""
		Save some statistics for eval using just one batch.
		"""
		stat_pred = self.policy(obs).detach()
		accuracy = th.eq(stat_pred.argmax(dim=1),actions.argmax(dim=1)).float().mean()

		if self._need_to_update_eval_statistics:
			self._need_to_update_eval_statistics = False
			self.eval_statistics['BC Loss'] = np.mean(ptu.get_numpy(bc_loss))
			self.eval_statistics['BC Accuracy'] = np.mean(ptu.get_numpy(accuracy))

	def get_diagnostics(self):
		return self.eval_statistics

	def end_epoch(self, epoch):
		self._need_to_update_eval_statistics = True

	@property
	def networks(self):
		nets = [
			self.policy,
		]
		return nets

	def get_snapshot(self):
		return dict(
			policy = self.policy,
		)
