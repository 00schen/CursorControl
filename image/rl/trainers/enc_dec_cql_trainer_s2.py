import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
from .dqn_trainer import DQNTrainer

class EncDecCQLTrainer(DQNTrainer):
	def __init__(self,
			rf,
			encoder,
			recon_decoder,
			pred_logvar,
			qf,target_qf,
			optimizer,
			temp=1.0,
			min_q_weight=1.0,
			add_ood_term=-1,
			beta = 1,
			use_noise = True,
			use_supervised = 'none',
			**kwargs):
		super().__init__(qf,target_qf,optimizer,**kwargs)
		# self.qf_optimizer also optimizes encoder
		self.rf = rf
		self.encoder = encoder
		self.recon_decoder = recon_decoder
		self.pred_logvar = pred_logvar
		self.temp = temp
		self.min_q_weight = min_q_weight
		self.add_ood_term = add_ood_term
		self.beta = beta
		self.use_noise = use_noise
		self.use_supervised = use_supervised

	def train_from_torch(self, batch):
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']
		rewards = batch['rewards']
		episode_success = batch['episode_success']
		gt_goal = batch['curr_goal']
		inputs = batch['curr_gaze_features']

		loss = 0

		"""
		Supervised loss
		"""
		pred_goal = self.encoder(inputs)
		if 'target' in self.use_supervised:
			supervised_loss = F.mse_loss(pred_goal,gt_goal)
			loss += supervised_loss

		if self.use_noise:
			noisy_goal = pred_goal+th.normal(ptu.zeros(pred_goal.shape),1)*th.exp(self.pred_logvar/2)
			# gaze_kl_loss = 0.5 * (1 + (pred_logvar-prior_logvar) - th.square(pred_goal-prior_pos) - (pred_logvar-prior_logvar).exp()).sum(dim=2).mean()
		else:
			noisy_goal = pred_goal
		# loss += self.beta*(gaze_kl_loss)
		
		curr_obs_features = [obs,noisy_goal]
		next_obs_features = [next_obs,noisy_goal]

		"""
		Rf loss
		"""
		if 'recon' in self.use_supervised:
			recon = self.recon_decoder(*curr_obs_features)
			recon_loss = F.mse_loss(recon,real_obs_features[0])
			loss += recon_loss
		if 'success' in self.use_supervised:
			pred_success = self.rf(*curr_obs_features,*next_obs_features)
			accuracy = th.eq(episode_success.bool(),F.sigmoid(pred_success.detach())>.5).float().mean()
			rf_loss = F.binary_cross_entropy_with_logits(pred_success.flatten(),episode_success.flatten())
			loss += rf_loss

		"""
		Q loss
		"""
		if 'bellman' in self.use_supervised:
			best_action_idxs = self.qf(*next_obs_features).max(
				1, keepdim=True
			)[1]
			target_q_values = self.target_qf(*next_obs_features).gather(
												1, best_action_idxs
											)
			y_target = rewards + (1. - terminals) * self.discount * target_q_values
			y_target = y_target.detach()

			# actions is a one-hot vector
			curr_qf = self.qf(*curr_obs_features)
			y_pred = th.sum(curr_qf * actions, dim=1, keepdim=True)
			loss += self.qf_criterion(y_pred, y_target)

		"""CQL term"""
		# min_qf_loss = th.logsumexp(curr_qf / self.temp, dim=1,).mean() * self.temp
		# min_qf_loss = min_qf_loss - y_pred.mean()

		# if self.add_ood_term < 0 or self._n_train_steps_total < self.add_ood_term:
		# 	loss += min_qf_loss * self.min_q_weight

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
			# self.eval_statistics['QF OOD Loss'] = np.mean(ptu.get_numpy(min_qf_loss))
			# self.eval_statistics['RF Accuracy'] = np.mean(ptu.get_numpy(accuracy))
			# self.eval_statistics['Predicted Logvar'] = np.mean(ptu.get_numpy(user_logvar))
			if 'bellman' in self.use_supervised:
				self.eval_statistics.update(create_stats_ordered_dict(
					'Q Predictions',
					ptu.get_numpy(y_pred),
				))

	@property
	def networks(self):
		nets = [
			self.rf,
			self.encoder,
			self.recon_decoder,
			self.qf,
			self.target_qf,
		]
		return nets

	def get_snapshot(self):
		return dict(
			rf = self.rf,
			encoder = self.encoder,
			recon_decoder=self.recon_decoder,
			qf = self.qf,
			target_qf = self.target_qf,
		)
