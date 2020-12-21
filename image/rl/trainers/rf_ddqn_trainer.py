import torch.optim as optim
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer

class RfDDQNTrainer(DoubleDQNTrainer):
	def __init__(self,qf1,qf2,target_qf1,target_qf2,rf,**kwargs):
		super().__init__(qf1,target_qf1,**kwargs)
		self.qf1 = self.qf
		self.target_qf1 = self.target_qf
		self.qf2 = qf2
		self.target_qf2 = target_qf2
		self.rf = rf
		self.qf1_optimizer = self.qf_optimizer
		self.qf2_optimizer = optim.Adam(
			self.qf2.parameters(),
			lr=self.learning_rate,
		)	
		self.rf_optimizer = optim.Adam(
			self.rf.parameters(),
			lr=self.learning_rate,
		)		

	def pretrain_rf(self,batch):
		noop = th.minimum(batch['rewards']+1,0)
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		pred_reward = self.rf(obs,next_obs)*actions
		rf_loss = F.binary_cross_entropy_with_logits(pred_reward,noop.Long())

		self.rf_optimizer.zero_grad()
		rf_loss.backward()
		self.rf_optimizer.step()

	def train_from_torch(self, batch):
		noop = th.minimum(batch['rewards']+1,0)
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		"""
		Reward and R loss
		"""
		pred_reward = self.rf(obs,next_obs)*actions
		rewards = pred_reward.clone().detach()
		rf_loss = F.binary_cross_entropy_with_logits(pred_reward,noop.Long())

		"""
		Q loss
		"""
		best_action_idxs = th.min(self.qf1(next_obs),self.qf2(next_obs)).max(
			1, keepdim=True
		)[1]
		target_q_values = th.min(self.target_qf1(next_obs).gather(
											1, best_action_idxs
										),
									self.target_qf2(next_obs).gather(
											1, best_action_idxs
										)
								)
		y_target = rewards + (1. - terminals) * self.discount * target_q_values
		y_target = y_target.detach()
		# actions is a one-hot vector
		y1_pred = th.sum(self.qf1(obs) * actions, dim=1, keepdim=True)
		qf1_loss = self.qf_criterion(y1_pred, y_target)
		y2_pred = th.sum(self.qf2(obs) * actions, dim=1, keepdim=True)
		qf2_loss = self.qf_criterion(y2_pred, y_target)	

		"""
		Update Q networks
		"""
		self.rf_optimizer.zero_grad()
		rf_loss.backward()
		self.rf_optimizer.step()
		self.qf1_optimizer.zero_grad()
		qf1_loss.backward()
		self.qf1_optimizer.step()
		self.qf2_optimizer.zero_grad()
		qf2_loss.backward()
		self.qf2_optimizer.step()

		"""
		Soft target network updates
		"""
		if self._n_train_steps_total % self.target_update_period == 0:
			ptu.soft_update_from_to(
				self.qf1, self.target_qf1, self.soft_target_tau
			)
			ptu.soft_update_from_to(
				self.qf2, self.target_qf2, self.soft_target_tau
			)

		"""
		Save some statistics for eval using just one batch.
		"""
		if self._need_to_update_eval_statistics:
			self._need_to_update_eval_statistics = False
			self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
			self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q1 Predictions',
				ptu.get_numpy(y1_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q2 Predictions',
				ptu.get_numpy(y2_pred),
			))

	@property
	def networks(self):
		nets = [
			self.qf1,
			self.target_qf1,
			self.qf2,
			self.target_qf2,
		]
		return nets

	def get_snapshot(self):
		return dict(
			qf1 = self.qf1,
			target_qf1 = self.target_qf1,
			qf2 = self.qf2,
			target_qf2 = self.target_qf2,
		)
