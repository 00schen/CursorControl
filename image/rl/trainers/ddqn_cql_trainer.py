import torch as th
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import OrderedDict

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp
# from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core import logger

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

class DDQNCQLTrainer(DQNTrainer):
	def __init__(self,qf,target_qf,
			optimizer,
			temp=1.0,
			min_q_weight=1.0,
			reward_update_period=.1,
			add_ood_term=-1,
			alpha = .4,
			**kwargs):
		super().__init__(qf,target_qf,optimizer,**kwargs)
		self.temp = temp
		self.min_q_weight = min_q_weight
		self.add_ood_term = add_ood_term
		self.alpha = alpha

	def train_from_torch(self, batch):
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']
		# gaze = batch['gaze'].flatten()

		"""
		Reward and R loss
		"""
		rewards = batch['rewards']

		# eps = th.normal(mean=ptu.zeros((obs.size(0), self.qf.embedding_dim)))
		"""
		Q loss
		"""
		# best_action_idxs = self.qf(next_obs,eps)[1].max(
		best_action_idxs = self.qf(next_obs).max(
			1, keepdim=True
		)[1]
		# target_q_values = self.target_qf(next_obs,eps)[1].gather(
		target_q_values = self.target_qf(next_obs).gather(
											1, best_action_idxs
										)
		y_target = rewards + (1. - terminals) * self.discount * target_q_values
		y_target = y_target.detach()
		
		# actions is a one-hot vector
		# kl_loss,curr_qf = self.qf(obs,eps)[:2]
		curr_qf = self.qf(obs)
		y_pred = th.sum(curr_qf * actions, dim=1, keepdim=True)
		# loss = self.qf_criterion(y_pred, y_target) + self.kl_weight*kl_loss
		loss = self.qf_criterion(y_pred, y_target)

		"""CQL term"""
		min_qf_loss = th.logsumexp(curr_qf / self.temp, dim=1,).mean() * self.temp
		min_qf_loss = min_qf_loss - y_pred.mean()

		if self.add_ood_term < 0 or self._n_train_steps_total < self.add_ood_term:
			loss += min_qf_loss * self.min_q_weight

		"""
		Update Q networks
		"""
		self.qf_optimizer.zero_grad()
		loss.backward()
		self.qf_optimizer.step()

		"""
		Soft target network updates
		"""
		if self._n_train_steps_total % self.target_update_period == 0:
			ptu.soft_update_from_to(
				self.qf, self.target_qf, self.soft_target_tau
			)

		"""
		Save some statistics for eval using just one batch.
		"""
		if self._need_to_update_eval_statistics:
			self._need_to_update_eval_statistics = False
			self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(loss))
			self.eval_statistics['QF OOD Loss'] = np.mean(ptu.get_numpy(min_qf_loss))
			self.eval_statistics['Noop Rate'] = np.mean(ptu.get_numpy(batch['noop']))
			self.eval_statistics.update(create_stats_ordered_dict(
				'R Predictions',
				ptu.get_numpy(rewards),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q Predictions',
				ptu.get_numpy(y_pred),
			))

	@property
	def networks(self):
		nets = [
			# self.rf,
			self.qf,
			self.target_qf,
		]
		return nets

	def get_snapshot(self):
		return dict(
			# rf =self.rf,
			qf = self.qf,
			target_qf = self.target_qf,
		)
