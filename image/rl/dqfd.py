import numpy as np
import torch
import torch.nn.functional as F

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.torch.dqn.double_dqn import DoubleDQNTrainer
from railrl.torch.core import np_to_pytorch_batch
from railrl.core import logger
from railrl.core.logging import add_prefix
import time

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
class DQfDReplayBuffer(EnvReplayBuffer):
	def __init__(self,max_replay_buffer_size,env,env_info_sizes=None):
		super().__init__(max_replay_buffer_size,env,env_info_sizes)
		self.demos_replay_buffer = EnvReplayBuffer(max_replay_buffer_size,env,env_info_sizes)

	def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
		if kwargs['env_info'].get('offline',False):
			self.demos_replay_buffer.add_sample(observation, action, reward, terminal, next_observation, **kwargs)
		else:
			super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)

	def random_batch(self, batch_size):
		if self._size >= int(batch_size*.25):
			online_batch = super().random_batch(int(batch_size*.25))
			demo_batch = self.demos_replay_buffer.random_batch(batch_size-int(batch_size*.25))
		else:
			online_batch = super().random_batch(self._size)
			demo_batch = self.demos_replay_buffer.random_batch(batch_size-self._size)
		demo_batch1 = self.demos_replay_buffer.random_batch(batch_size)
		batch = dict(
			observations=np.array((np.concatenate((online_batch['observations'],demo_batch['observations'])),demo_batch1['observations'])),
			actions=np.array((np.concatenate((online_batch['actions'],demo_batch['actions'])),demo_batch1['actions'])),
			rewards=np.array((np.concatenate((online_batch['rewards'],demo_batch['rewards'])),demo_batch1['rewards'])),
			terminals=np.array((np.concatenate((online_batch['terminals'],demo_batch['terminals'])),demo_batch1['terminals'])),
			next_observations=np.array((np.concatenate((online_batch['next_observations'],demo_batch['next_observations'])),demo_batch1['next_observations'])),
		)
		return batch

	def num_steps_can_sample(self):
		return self.demos_replay_buffer.num_steps_can_sample()

class DQfDTrainer(DoubleDQNTrainer):
	def __init__(
			self,
			qf,
			target_qf,
			lam1=.1,
			lam2=1e-5,
			margin=.8,
			pretrain_steps=int(5e4),
			**kwargs
	):
		super().__init__(qf,target_qf,**kwargs)
		self.lam1 = lam1
		self.lam2 = lam2
		self.pretrain_steps = pretrain_steps
		self.margin = margin

	def pretrain(self):
		prev_time = time.time()
		for i in range(self.pretrain_steps):
			self.eval_statistics = dict()
			if i % 1000 == 0:
				self._need_to_update_eval_statistics=True
			train_data = self.replay_buffer.random_batch(128)
			train_data = np_to_pytorch_batch(train_data)
			self.train_from_torch(train_data)

			if i % 1000 == 0:
				self.eval_statistics["batch"] = i
				self.eval_statistics["epoch_time"] = time.time()-prev_time
				stats_with_prefix = add_prefix(self.eval_statistics, prefix="trainer/")
				logger.record_dict(stats_with_prefix)
				logger.dump_tabular(with_prefix=True, with_timestamp=False)
				prev_time = time.time()

		self._need_to_update_eval_statistics = True
		self.eval_statistics = dict()

	def train_from_torch(self, batch):
		rewards,rewards1 = batch['rewards']
		terminals,terminals1 = batch['terminals']
		obs,obs1 = batch['observations']
		actions,actions1 = batch['actions']
		next_obs,next_obs1 = batch['next_observations']

		"""
		Compute loss
		"""

		best_action_idxs = self.qf(next_obs).max(
			1, keepdim=True
		)[1]
		target_q_values = self.target_qf(next_obs).gather(
			1, best_action_idxs
		).detach()
		y_target = rewards + (1. - terminals) * self.discount * target_q_values
		y_target = y_target.detach()
		# actions is a one-hot vector
		y_pred = torch.sum(self.qf(obs) * actions, dim=1, keepdim=True)
		qf_loss = self.qf_criterion(y_pred, y_target)

		action_dim = len(actions[0])
		margin = (self.margin/2*(F.one_hot(torch.arange(0,action_dim).repeat(len(actions1),1),action_dim)!=actions1.cpu().unsqueeze(1)).sum(dim=2)\
				+self.qf(next_obs1).cpu()).max(1, keepdim=True)[0]
		pred_target_q = self.target_qf(next_obs).gather(
			1, actions1.argmax(1,keepdim=True)
		).detach().cpu()
		margin_loss = self.lam1*F.l1_loss(margin,pred_target_q)

		reg_loss = 0
		for param in self.qf.parameters():
			reg_loss += param.detach().square().sum()
		reg_loss *= self.lam2

		loss = qf_loss + margin_loss + reg_loss

		"""
		Update networks
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
			self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Y Predictions',
				ptu.get_numpy(y_pred),
			))

	@property
	def networks(self):
		nets = [
			self.qf,
			self.target_qf,
		]
		return nets