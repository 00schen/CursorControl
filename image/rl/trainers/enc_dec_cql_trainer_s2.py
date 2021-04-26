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
			gt_encoder,
			encoder,
			# gt_logvar,
			recon_decoder,
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
			use_supervised = 'none',
			gt_prior = True,
			**kwargs):
		super().__init__(qf,target_qf,optimizer,**kwargs)
		# self.qf_optimizer also optimizes encoder
		self.rf = rf
		self.gt_encoder = gt_encoder
		self.encoder = encoder
		self.recon_decoder = recon_decoder
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
		self.gt_prior = gt_prior

	def train_from_torch(self, batch):
		# terminals = batch['terminals'].repeat(2,1)
		# actions = batch['actions'].repeat(2,1)
		# obs = batch['observations'].repeat(2,1)
		# next_obs = batch['next_observations'].repeat(2,1)
		# rewards = batch['rewards'].repeat(2,1)
		# gt_target_pos = th.cat((batch['current_target'],ptu.zeros((batch['current_target'].shape[0],self.encoder.input_size-batch['current_target'].shape[1]))),dim=1)
		# target_pos = batch['curr_goal']
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
		real_obs_features = [batch[key] for key in batch.keys() if 'curr_' in key and 'goal' not in key]
		user_latent = self.encoder(*real_obs_features)
		user_mean,user_logvar = user_latent[:,:3],user_latent[:,3:]
		user_logvar = self.gaze_logvar if self.global_noise else user_logvar
		if self.use_supervised == 'target':
			gt_latent = self.gt_encoder(target_pos)
			gt_mean, gt_logvar = gt_latent[:,:3],gt_latent[:,3:]
			supervised_loss = F.mse_loss(user_mean,gt_mean)
			loss += supervised_loss

		# (prior_pos,prior_logvar) = (gt_mean.mean(dim=0,keepdim=True).detach(),gt_logvar.mean(dim=0,keepdim=True).detach()) if self.gt_prior else (0,0)
		(prior_pos,prior_logvar) =  (0,0)
		if self.use_gaze_noise:
			noisy_user_pos = user_mean+th.normal(ptu.zeros(user_mean.shape),1)*th.exp(user_logvar/2)
			# noisy_target_pos = gt_mean+th.normal(ptu.zeros(user_mean.shape),1)*th.exp(gt_logvar/2)
			# gt_kl_loss = -0.5 * (1 + gt_logvar - th.square(gt_mean) - gt_logvar.exp()).sum(dim=1).mean()
			user_kl_loss = -0.5 * (1 + (user_logvar-prior_logvar) - th.square(user_mean-prior_pos) - (user_logvar-prior_logvar).exp()).sum(dim=1).mean()
			
			# loss += self.beta*(gt_kl_loss + gaze_th.square(user_mean-prior_pos) kl_loss)
		else:
			noisy_user_pos = user_mean
			user_kl_loss = -0.5 * th.square(user_mean-prior_pos).mean()
		loss += self.beta*(user_kl_loss)
		
		# curr_obs_features = [obs,th.cat((noisy_user_pos,noisy_target_pos))]
		# next_obs_features = [next_obs,th.cat((noisy_user_pos,noisy_target_pos))]
		curr_obs_features = [obs,noisy_user_pos]
		next_obs_features = [next_obs,noisy_user_pos]

		"""
		Rf loss
		"""
		if 'recon' in self.use_supervised:
			recon = self.recon_decoder(noisy_user_pos)
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
			self.eval_statistics['Predicted Logvar'] = np.mean(ptu.get_numpy(user_logvar))
			# self.eval_statistics['Prior Logvar'] = np.mean(ptu.get_numpy(prior_logvar))
			# self.eval_statistics['Predicted Mean'] = np.mean(ptu.get_numpy(gt_mean))
			# self.eval_statistics['Prior Mean'] = np.mean(ptu.get_numpy(prior_pos))
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
			self.recon_decoder,
			self.qf,
			self.target_qf,
		]
		return nets

	def get_snapshot(self):
		return dict(
			rf = self.rf,
			# gt_encoder = self.gt_encoder,
			encoder = self.encoder,
			recon_decoder=self.recon_decoder,
			# gt_noise = self.gt_logvar,
			gaze_noise = self.gaze_logvar,
			qf = self.qf,
			target_qf = self.target_qf,
		)
