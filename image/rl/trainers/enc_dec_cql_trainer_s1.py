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

class EncDecCQLTrainer(DQNTrainer):
	def __init__(self,
			rf,
			# gt_encoder,
			encoder,
			# gt_logvar,
			gaze_logvar,
			qf,target_qf,
			optimizer,
			temp=1.0,
			min_q_weight=1.0,
			add_ood_term=-1,
			alpha = .4,
			beta = 1,
			use_gaze_noise = False,
			global_noise = True,
			use_supervised = 'False',
			**kwargs):
		super().__init__(qf,target_qf,optimizer,**kwargs)
		# self.qf_optimizer also optimizes encoder
		self.rf = rf
		# self.gt_encoder = gt_encoder
		self.encoder = encoder
		# self.gt_logvar = gt_logvar
		self.gaze_logvar = gaze_logvar
		self.temp = temp
		self.min_q_weight = min_q_weight
		self.add_ood_term = add_ood_term
		self.alpha = alpha
		self.beta = beta
		self.use_gaze_noise = use_gaze_noise
		self.global_noise = global_noise
		self.use_supervised = use_supervised

	def train_from_torch(self, batch):
		# terminals = batch['terminals'].repeat(2,1)
		# actions = batch['actions'].repeat(2,1)
		# obs = batch['observations'].repeat(2,1)
		# next_obs = batch['next_observations'].repeat(2,1)
		# rewards = batch['rewards'].repeat(2,1)
		# target_pos = th.cat((batch['current_target'],ptu.zeros((batch['current_target'].shape[0],self.encoder.input_size-batch['current_target'].shape[1]))),dim=1)
		# episode_success = batch['episode_success'].repeat(2,1)
		terminals = batch['terminals']
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']
		rewards = batch['rewards']
		episode_success = batch['episode_success']

		loss = 0

		"""
		Supervised loss
		"""
		real_obs_features = [batch[key] for key in batch.keys() if 'curr_' in key]
		pred_pos = self.encoder(*real_obs_features)
		pred_pos,pred_logvar = pred_pos[:,:3],pred_pos[:,3:]
		pred_logvar = self.gaze_logvar if self.global_noise else pred_logvar
		# supervised_loss = F.mse_loss(pred_pos,target_pos)
		# loss = supervised_loss
		# pred_target_pos = self.gt_encoder(target_pos)

		# noisy_target_pos = th.normal(pred_target_pos,th.exp(self.gt_logvar/2))
		if self.use_gaze_noise:
			noisy_pred_pos = pred_pos+th.normal(ptu.zeros(pred_pos.shape),1)*th.exp(pred_logvar/2)
			# gt_kl_loss = -0.5 * (1 + self.gt_logvar - th.square(pred_target_pos) - self.gt_logvar.exp()).sum(dim=1).mean()
			gaze_kl_loss = -0.5 * (1 + pred_logvar - th.square(pred_pos) - pred_logvar.exp()).sum(dim=1).mean()
			loss += self.beta*(gaze_kl_loss)
			# loss += self.beta*(gt_kl_loss + gaze_kl_loss)
		else:
			noisy_pred_pos = pred_pos
		
		# curr_obs_features = [obs,th.cat((noisy_pred_pos,noisy_target_pos))]
		# next_obs_features = [next_obs,th.cat((noisy_pred_pos,noisy_target_pos))]
		curr_obs_features = [obs,noisy_pred_pos]
		next_obs_features = [next_obs,noisy_pred_pos]


		"""
		Rf loss
		"""
		if 'success' in self.use_supervised:
			pred_success = self.rf(*curr_obs_features,*next_obs_features)
			rf_loss = F.binary_cross_entropy_with_logits(pred_success.flatten(),episode_success.flatten())
			loss += rf_loss
			accuracy = th.eq(episode_success.bool(),F.sigmoid(pred_success.detach())>.5).float().mean()

		"""
		Q loss
		"""
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
			# self.eval_statistics['RF Accuracy'] = np.mean(ptu.get_numpy(accuracy))
			self.eval_statistics['Predicted Logvar'] = np.mean(ptu.get_numpy(pred_logvar))			
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q Predictions',
				ptu.get_numpy(y_pred),
			))

	@property
	def networks(self):
		nets = [
			self.rf,
			# self.gt_encoder,
			self.encoder,
			self.qf,
			self.target_qf,
		]
		return nets

	def get_snapshot(self):
		return dict(
			rf = self.rf,
			# gt_encoder = self.gt_encoder,
			encoder = self.encoder,
			# gt_noise = self.gt_logvar,
			gaze_noise = self.gaze_logvar,
			qf = self.qf,
			target_qf = self.target_qf,
		)
