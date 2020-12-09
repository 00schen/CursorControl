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

class DDQNCQLTrainer(DoubleDQNTrainer):
	def __init__(self,qf1,qf2,target_qf1,target_qf2,
			temp=1.0,
            min_q_weight=1.0,
			**kwargs):
		
		super().__init__(qf1,target_qf1,**kwargs)
		self.qf1 = self.qf
		self.target_qf1 = self.target_qf
		self.qf2 = qf2
		self.target_qf2 = target_qf2
		self.qf1_optimizer = self.qf_optimizer
		self.qf2_optimizer = optim.Adam(
			self.qf2.parameters(),
			lr=self.learning_rate,
		)
		self.temp = temp
		self.min_q_weight = min_q_weight

	def bc(self,batch):
		batch = np_to_pytorch_batch(batch)
		actions = batch['actions']
		concat_obs = batch['observations']

		target = actions.argmax(dim=1).squeeze()
		qf1_loss = F.cross_entropy(self.qf1(concat_obs),target)
		qf2_loss = F.cross_entropy(self.qf2(concat_obs),target)

		self.qf1_optimizer.zero_grad()
		qf1_loss.backward()
		self.qf1_optimizer.step()
		self.qf2_optimizer.zero_grad()
		qf2_loss.backward()
		self.qf2_optimizer.step()

		ptu.soft_update_from_to(
			self.qf1, self.target_qf1, self.soft_target_tau
		)
		ptu.soft_update_from_to(
			self.qf2, self.target_qf2, self.soft_target_tau
		)

	def train_from_torch(self, batch):
		rewards = batch['rewards']
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

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
		curr_qf1 = self.qf1(obs)
		curr_qf2  = self.qf2(obs)
		y1_pred = th.sum(curr_qf1 * actions, dim=1, keepdim=True)
		qf1_loss = self.qf_criterion(y1_pred, y_target)
		y2_pred = th.sum(curr_qf2 * actions, dim=1, keepdim=True)
		qf2_loss = self.qf_criterion(y2_pred, y_target)	

		"""CQL term"""
		min_qf1_loss = th.logsumexp(curr_qf1 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
		min_qf2_loss = th.logsumexp(curr_qf2 / self.temp, dim=1,).mean() * self.min_q_weight * self.temp
		min_qf1_loss = min_qf1_loss - curr_qf1.mean() * self.min_q_weight
		min_qf2_loss = min_qf2_loss - curr_qf2.mean() * self.min_q_weight

		qf1_loss += min_qf1_loss
		qf2_loss += min_qf2_loss

		"""
		Update Q networks
		"""
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
			self.eval_statistics['QF1 OOD Loss'] = np.mean(ptu.get_numpy(qf1_loss))
			self.eval_statistics['QF2 OOD Loss'] = np.mean(ptu.get_numpy(qf2_loss))
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
