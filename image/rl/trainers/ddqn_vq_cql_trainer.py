import torch as th
import torch.nn.functional as F
import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
from rl.trainers.ddqn_cql_trainer import DDQNCQLTrainer

class DDQNVQCQLTrainer(DDQNCQLTrainer):
	def train_from_torch(self, batch):
		noop = th.clamp(batch['rewards']+1,0,1)
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		"""
		Reward and R loss
		"""
		if not self.ground_truth:
			rewards = (1-self.rf(obs,next_obs).exp()).log()*-1*batch['rewards']
			if self._n_train_steps_total % self.reward_update_period == 0:
				noop_prop = noop.mean().item()
				noop_prop = max(1e-4,1-noop_prop)/max(1e-4,noop_prop)
				rf_obs,rf_next_obs,rf_noop = self.mixup(obs,next_obs,noop)
				pred_reward = self.rf(rf_obs,rf_next_obs)
				rf_loss = F.binary_cross_entropy_with_logits(pred_reward,rf_noop,pos_weight=ptu.tensor([noop_prop]))

				self.rf_optimizer.zero_grad()
				rf_loss.backward()
				self.rf_optimizer.step()
		else:
			rewards = batch['rewards']

		"""
		Q loss
		"""
		best_action_idxs = self.qf(next_obs, skip_encoder=self.latent_train).max(
			1, keepdim=True
		)[1]

		target_q_values = self.target_qf(next_obs, skip_encoder=self.latent_train).gather(
											1, best_action_idxs
										)
		y_target = rewards + (1. - terminals) * self.discount * target_q_values
		y_target = y_target.detach()
		# actions is a one-hot vector

		curr_qf, vq_loss1, vq_loss2 = self.qf(obs, return_vq=True, skip_encoder=self.latent_train)
		y_pred = th.sum(curr_qf * actions, dim=1, keepdim=True)
		qf_loss = self.qf_criterion(y_pred, y_target)
		if not self.latent_train:
			qf_loss = qf_loss + self.beta * 2 * vq_loss1 + self.beta * vq_loss2

		if not self.latent_train and hasattr(self.qf, 'rew_classification'):
			num_pos = th.sum(terminals)
			num_neg = terminals.size(0) - num_pos
			rew_class, rew_vq_loss1, rew_vq_loss2 = self.qf.rew_classification(next_obs, return_vq=True,
																			   train_encoder=self.train_encoder_on_rew_class)
			rew_class_loss = th.nn.BCEWithLogitsLoss(pos_weight=num_neg / (num_pos + 1e-6))(rew_class, terminals)
			qf_loss += self.rew_class_weight * rew_class_loss + self.beta * 2 * rew_vq_loss1 + self.beta + vq_loss2

		"""CQL term"""
		min_qf_loss = th.logsumexp(curr_qf / self.temp, dim=1,).mean() * self.temp
		min_qf_loss = min_qf_loss - y_pred.mean()

		if self.add_ood_term < 0 or self._n_train_steps_total < self.add_ood_term:
			qf_loss += min_qf_loss * self.min_q_weight

		# qf_loss = th.nn.CrossEntropyLoss()(curr_qf, th.argmax(actions, dim=1))
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
			# self.eval_statistics['RF Loss'] = np.mean(ptu.get_numpy(rf_loss))
			self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
			self.eval_statistics['QF OOD Loss'] = np.mean(ptu.get_numpy(min_qf_loss))
			self.eval_statistics.update(create_stats_ordered_dict(
				'R Predictions',
				ptu.get_numpy(rewards),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q Predictions',
				ptu.get_numpy(y_pred),
			))
