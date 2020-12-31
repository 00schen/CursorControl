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
	def __init__(self,qf1,qf2,target_qf1,target_qf2,rf,
			temp=1.0,
            min_q_weight=1.0,
			reward_update_period=1,
			**kwargs):
		super().__init__(qf1,target_qf1,**kwargs)
		self.reward_update_period = reward_update_period
		self.temp = temp
		self.min_q_weight = min_q_weight
		self.qf1 = self.qf
		self.target_qf1 = self.target_qf
		self.qf2 = qf2
		self.target_qf2 = target_qf2
		self.rf = rf
		self.qf1_optimizer = self.qf_optimizer = optim.Adam(
			self.qf1.parameters(),
			lr=self.learning_rate,
			weight_decay=1e-5,
		)	
		self.qf2_optimizer = optim.Adam(
			self.qf2.parameters(),
			lr=self.learning_rate,
			weight_decay=1e-5,
		)	
		self.rf_optimizer = optim.Adam(
			self.rf.parameters(),
			lr=1e-3,
			weight_decay=1e-5,
		)

	def pretrain_rf(self,batch):
		batch = np_to_pytorch_batch(batch)
		noop = th.clamp(batch['rewards']+1,0,1)
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		noop_prop = noop.mean().item()
		noop_prop = max(1e-4,1-noop_prop)/noop_prop
		pred_reward = th.sum(self.rf(obs,next_obs)*actions, dim=1, keepdim=True)
		rf_loss = F.binary_cross_entropy_with_logits(pred_reward,noop,pos_weight=ptu.tensor([noop_prop]))

		self.rf_optimizer.zero_grad()
		rf_loss.backward()
		self.rf_optimizer.step()

	def train_from_torch(self, batch):
		noop = th.clamp(batch['rewards']+1,0,1)
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		"""
		Reward and R loss
		"""
		noop_prop = noop.mean().item()
		noop_prop = max(1e-4,1-noop_prop)/noop_prop
		pred_reward = th.sum(self.rf(obs,next_obs)*actions, dim=1, keepdim=True)
		rf_loss = F.binary_cross_entropy_with_logits(pred_reward,noop,pos_weight=ptu.tensor([noop_prop]))
		rewards = pred_reward.clone().detach()

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
		min_qf1_loss = th.logsumexp(curr_qf1 / self.temp, dim=1,).mean() * self.temp
		min_qf2_loss = th.logsumexp(curr_qf2 / self.temp, dim=1,).mean() * self.temp
		min_qf1_loss = min_qf1_loss - y1_pred.mean()
		min_qf2_loss = min_qf2_loss - y2_pred.mean()

		qf1_loss += min_qf1_loss * self.min_q_weight
		qf2_loss += min_qf2_loss * self.min_q_weight

		"""
		Update Q networks
		"""
		if self._n_train_steps_total % self.reward_update_period == 0:
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
			self.eval_statistics['RF Loss'] = np.mean(ptu.get_numpy(rf_loss))
			self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
			self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
			self.eval_statistics['QF1 OOD Loss'] = np.mean(ptu.get_numpy(qf1_loss))
			self.eval_statistics['QF2 OOD Loss'] = np.mean(ptu.get_numpy(qf2_loss))
			self.eval_statistics.update(create_stats_ordered_dict(
				'R Predictions',
				ptu.get_numpy(rewards),
			))
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
			self.rf,
			self.qf1,
			self.target_qf1,
			self.qf2,
			self.target_qf2,
		]
		return nets

	def get_snapshot(self):
		return dict(
			rf =self.rf,
			qf1 = self.qf1,
			target_qf1 = self.target_qf1,
			qf2 = self.qf2,
			target_qf2 = self.target_qf2,
		)
