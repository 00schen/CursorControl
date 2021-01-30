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
from rlkit.core import logger

class DDQNCQLTrainer(DoubleDQNTrainer):
	def __init__(self,qf1,target_qf1,rf,
			temp=1.0,
			min_q_weight=1.0,
			reward_update_period=.1,
			add_ood_term=-1,
			alpha = .4,
			ground_truth=False,
			**kwargs):
		super().__init__(qf1,target_qf1,**kwargs)
		self.reward_update_period = reward_update_period
		self.temp = temp
		self.min_q_weight = min_q_weight
		self.add_ood_term = add_ood_term
		self.alpha = alpha
		self.ground_truth = ground_truth
		self.qf1 = self.qf
		self.target_qf1 = self.target_qf
		self.rf = rf
		self.qf1_optimizer = self.qf_optimizer = optim.Adam(
			self.qf1.parameters(),
			lr=self.learning_rate,
			weight_decay=1e-5,
		)	
		self.rf_optimizer = optim.Adam(
			self.rf.parameters(),
			lr=1e-3,
			weight_decay=1e-5,
		)

	# def pretrain_rf(self,batch):
	# 	batch = np_to_pytorch_batch(batch)
	# 	noop = th.clamp(batch['rewards']+1,0,1)
	# 	actions = batch['actions']
	# 	obs = batch['observations']
	# 	next_obs = batch['next_observations']

	# 	noop_prop = noop.mean().item()
	# 	noop_prop = max(1e-4,1-noop_prop)/max(1e-4,noop_prop)

	# 	rf_obs,rf_next_obs,rf_noop = mixup(obs,next_obs,noop)
	# 	pred_reward = self.rf(rf_obs,rf_next_obs)
	# 	accuracy = th.eq((self.rf(obs,next_obs)>np.log(.5)).float(),noop).float().mean()
	# 	rf_loss = F.binary_cross_entropy_with_logits(pred_reward,rf_noop,pos_weight=ptu.tensor([noop_prop]))

	# 	self.rf_optimizer.zero_grad()
	# 	rf_loss.backward()
	# 	self.rf_optimizer.step()

	# 	rf_statistics = {}
	# 	rf_statistics['RF Loss'] = np.mean(ptu.get_numpy(rf_loss))
	# 	rf_statistics['RF Accuracy'] = np.mean(ptu.get_numpy(accuracy))
	# 	logger.record_dict(rf_statistics, prefix='')
	# 	logger.dump_tabular(with_prefix=False, with_timestamp=False)

	def train_from_torch(self, batch):
		noop = th.clamp(batch['rewards']+1,0,1)
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		"""
		Reward and R loss
		"""
		# if not self.ground_truth:
		# 	# rewards = (1-self.rf(obs,next_obs).exp()).log()*-1*batch['rewards']
		# 	# if self._n_train_steps_total % self.reward_update_period == 0:
		# 	# 	noop_prop = noop.mean().item()
		# 	# 	noop_prop = max(1e-4,1-noop_prop)/max(1e-4,noop_prop)
		# 	# 	rf_obs,rf_next_obs,rf_noop = self.mixup(obs,next_obs,noop)
		# 	# 	pred_reward = self.rf(rf_obs,rf_next_obs)
		# 	# 	rf_loss = F.binary_cross_entropy_with_logits(pred_reward,rf_noop,pos_weight=ptu.tensor([noop_prop]))

		# 	# 	self.rf_optimizer.zero_grad()
		# 	# 	rf_loss.backward()
		# 	# 	self.rf_optimizer.step()
		# else:
		# 	rewards = batch['rewards']
		rewards = batch['rewards']

		"""
		Q loss
		"""
		best_action_idxs = self.qf1(next_obs).max(
			1, keepdim=True
		)[1]
		target_q_values = self.target_qf1(next_obs).gather(
											1, best_action_idxs
										)
		y_target = rewards + (1. - terminals) * self.discount * target_q_values
		y_target = y_target.detach()
		# actions is a one-hot vector
		curr_qf1 = self.qf1(obs)
		y1_pred = th.sum(curr_qf1 * actions, dim=1, keepdim=True)
		qf1_loss = self.qf_criterion(y1_pred, y_target)

		"""CQL term"""
		min_qf1_loss = th.logsumexp(curr_qf1 / self.temp, dim=1,).mean() * self.temp
		min_qf1_loss = min_qf1_loss - y1_pred.mean()

		if self.add_ood_term < 0 or self._n_train_steps_total < self.add_ood_term:
			qf1_loss += min_qf1_loss * self.min_q_weight

		"""
		Update Q networks
		"""
		self.qf1_optimizer.zero_grad()
		qf1_loss.backward()
		self.qf1_optimizer.step()

		"""
		Soft target network updates
		"""
		if self._n_train_steps_total % self.target_update_period == 0:
			ptu.soft_update_from_to(
				self.qf1, self.target_qf1, self.soft_target_tau
			)

		"""
		Save some statistics for eval using just one batch.
		"""
		if self._need_to_update_eval_statistics:
			self._need_to_update_eval_statistics = False
			# self.eval_statistics['RF Loss'] = np.mean(ptu.get_numpy(rf_loss))
			self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
			self.eval_statistics['QF1 OOD Loss'] = np.mean(ptu.get_numpy(min_qf1_loss))
			self.eval_statistics.update(create_stats_ordered_dict(
				'R Predictions',
				ptu.get_numpy(rewards),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q1 Predictions',
				ptu.get_numpy(y1_pred),
			))

	@property
	def networks(self):
		nets = [
			self.rf,
			self.qf1,
			self.target_qf1,
		]
		return nets

	def get_snapshot(self):
		return dict(
			rf =self.rf,
			qf1 = self.qf1,
			target_qf1 = self.target_qf1,
		)
