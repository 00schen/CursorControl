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
	def __init__(self, qf, target_qf,
			temp=1.0,
            min_q_weight=1.0,
			reward_update_period=1,
			**kwargs):
		
		super().__init__(qf,target_qf,**kwargs)
		self.temp = temp
		self.min_q_weight = min_q_weight

	def pretrain_rf(self,batch):
		batch = np_to_pytorch_batch(batch)
		noop = th.clamp(batch['rewards']+1,0,1)
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		target = actions.argmax(dim=1).squeeze()
		qf_loss = F.cross_entropy(self.qf(concat_obs),target)

		self.qf_optimizer.zero_grad()
		qf_loss.backward()
		self.qf_optimizer.step()

		ptu.soft_update_from_to(
			self.qf, self.target_qf, self.soft_target_tau
		)

	def train_from_torch(self, batch):
		# noop = th.clamp(batch['rewards']+1,0,1)
		rewards = batch['rewards']
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		"""
		Reward and R loss
		"""
		# noop_prop = noop.mean().item()
		# noop_prop = max(1e-4,1-noop_prop)/noop_prop
		# pred_reward = th.sum(self.qf(obs,next_obs)*actions, dim=1, keepdim=True)
		# rf_loss = F.binary_cross_entropy_with_logits(pred_reward,noop,pos_weight=ptu.tensor([noop_prop]))
		# rewards = pred_reward.clone().detach()

		"""
		Q loss
		"""
		best_action_idxs = self.qf(next_obs).max(
			1, keepdim=True
		)[1]
		target_q_values = self.target_qf(next_obs).gather(1, best_action_idxs)

		y_target = rewards + (1. - terminals) * self.discount * target_q_values
		y_target = y_target.detach()
		# actions is a one-hot vector
		curr_qf = self.qf(obs)
		y_pred = th.sum(curr_qf * actions, dim=1, keepdim=True)
		qf_loss = self.qf_criterion(y_pred, y_target)

		"""CQL term"""
		min_qf_loss = (th.logsumexp(curr_qf / self.temp, dim=1, keepdim=True) * self.temp
						- y_pred).mean() * self.min_q_weight

		qf_loss += min_qf_loss

		"""
		Update Q networks
		"""
		self.qf_optimizer.zero_grad()
		qf_loss.backward()
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
			self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
			self.eval_statistics['QF OOD Loss'] = np.mean(ptu.get_numpy(qf_loss))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q Predictions',
				ptu.get_numpy(y_pred),
			))

	@property
	def networks(self):
		nets = [
			self.qf,
			self.target_qf,
		]
		return nets

	def get_snapshot(self):
		return dict(
			qf=self.qf,
			target_qf=self.target_qf,
		)
