import torch as th
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.core import logger

class DDQNCQLTrainer(DoubleDQNTrainer):
	def __init__(self,qf,target_qf,rf,
			temp=1.0,
			min_q_weight=1.0,
			reward_update_period=.1,
			add_ood_term=-1,
			alpha = .4,
			ground_truth=False,
			target_name='noop',
			**kwargs):
		super().__init__(qf,target_qf,**kwargs)
		self.reward_update_period = reward_update_period
		self.temp = temp
		self.min_q_weight = min_q_weight
		self.add_ood_term = add_ood_term
		self.alpha = alpha
		self.ground_truth = ground_truth
		self.target_name = target_name
		self.rf = rf

	def train_from_torch(self, batch):
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		"""
		Reward and R loss
		"""
		if not self.ground_truth:
			self.rf.train(False)
			r_pred = self.rf(obs,next_obs).detach()
			rewards = F.logsigmoid(r_pred)
			accuracy = th.eq(r_pred.sigmoid()>.5, batch['episode_success']).float().mean()
			if self._need_to_update_eval_statistics:
				self.eval_statistics['RF Accuracy'] = np.mean(ptu.get_numpy(accuracy))
		else:
			# rewards = batch['rewards']
			rewards = batch[self.target_name]-1
			# rewards = batch['noop']-1

		"""
		Q loss
		"""
		best_action_idxs = self.qf(next_obs).max(
			1, keepdim=True
		)[1]
		target_q_values = self.target_qf(next_obs).gather(
											1, best_action_idxs
										)
		y_target = rewards + (1. - terminals) * self.discount * target_q_values
		y_target = y_target.detach()
		
		# actions is a one-hot vector
		curr_qf = self.qf(obs)
		y_pred = th.sum(curr_qf * actions, dim=1, keepdim=True)
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
			self.rf,
			self.qf,
			self.target_qf,
		]
		return nets

	def get_snapshot(self):
		return dict(
			rf =self.rf,
			qf = self.qf,
			target_qf = self.target_qf,
		)
