import torch.optim as optim
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from itertools import chain

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
			beta=1,
			rew_class_weight=1,
			sample=False,
			latent_train=False,
			train_encoder_on_rew_class=True,
			freeze_decoder=False,
			train_qf_head=True,
			**kwargs):
		super().__init__(qf,target_qf,**kwargs)
		self.reward_update_period = reward_update_period
		self.temp = temp
		self.min_q_weight = min_q_weight
		self.add_ood_term = add_ood_term
		self.alpha = alpha
		self.ground_truth = ground_truth
		self.rf = rf
		self.rf_optimizer = optim.Adam(
			self.rf.parameters(),
			lr=1e-3,
			weight_decay=1e-5,
		)
		self.beta = beta
		self.rew_class_weight = rew_class_weight
		self.sample = sample
		self.latent_train = latent_train
		self.train_encoder_on_rew_class = train_encoder_on_rew_class

		params = self.qf.gaze_vae.encoder.parameters() if freeze_decoder else self.qf.parameters()
		if freeze_decoder and train_qf_head:
			params = chain(params, self.qf.decoder.last_fc.parameters())

		self.qf_optimizer = optim.Adam(
			params,
			lr=self.learning_rate,
		)

	def mixup(self,obs,next_obs,noop):
		lambd = np.random.beta(self.alpha, self.alpha, obs.size(0))
		lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
		lambd = obs.new(lambd)
		shuffle = th.randperm(obs.size(0)).to(obs.device)
		obs1,next_obs1,noop1 = obs[shuffle], next_obs[shuffle], noop[shuffle]

		obs1 = (obs * lambd.view(lambd.size(0),1) + obs1 * (1-lambd).view(lambd.size(0),1))
		next_obs1 = (next_obs * lambd.view(lambd.size(0),1) + next_obs1 * (1-lambd).view(lambd.size(0),1))
		noop1 = (noop * lambd.view(lambd.size(0),1) + noop1 * (1-lambd).view(lambd.size(0),1))
		return obs1,next_obs,noop

	def pretrain_rf(self,batch):
		batch = np_to_pytorch_batch(batch)
		noop = th.clamp(batch['rewards']+1,0,1)
		actions = batch['actions']
		obs = batch['observations']
		next_obs = batch['next_observations']

		noop_prop = noop.mean().item()
		noop_prop = max(1e-4,1-noop_prop)/max(1e-4,noop_prop)
		rf_obs,rf_next_obs,rf_noop = self.mixup(obs,next_obs,noop)
		pred_reward = self.rf(rf_obs,rf_next_obs)
		accuracy = th.eq((self.rf(obs,next_obs)>np.log(.5)).float(),noop).float().mean()
		rf_loss = F.binary_cross_entropy_with_logits(pred_reward,rf_noop,pos_weight=ptu.tensor([noop_prop]))

		self.rf_optimizer.zero_grad()
		rf_loss.backward()
		self.rf_optimizer.step()

		rf_statistics = {}
		rf_statistics['RF Loss'] = np.mean(ptu.get_numpy(rf_loss))
		rf_statistics['RF Accuracy'] = np.mean(ptu.get_numpy(accuracy))
		logger.record_dict(rf_statistics, prefix='')
		logger.dump_tabular(with_prefix=False, with_timestamp=False)

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
		eps = th.normal(mean=th.zeros((obs.size(0), self.qf.latent_dim)), std=1).to(ptu.device) if self.sample else None

		best_action_idxs = self.qf(next_obs, eps=eps, skip_encoder=self.latent_train).max(
			1, keepdim=True
		)[1]

		target_q_values = self.target_qf(next_obs, eps=eps, skip_encoder=self.latent_train).gather(
											1, best_action_idxs
										)
		y_target = rewards + (1. - terminals) * self.discount * target_q_values
		y_target = y_target.detach()
		# actions is a one-hot vector

		curr_qf, kl_loss = self.qf(obs, eps=eps, return_kl=True, skip_encoder=self.latent_train)
		# curr_qf = self.qf(obs)
		y_pred = th.sum(curr_qf * actions, dim=1, keepdim=True)
		qf_loss = self.qf_criterion(y_pred, y_target)
		if not self.latent_train:
			qf_loss = qf_loss + self.beta * kl_loss
		#qf_loss = self.qf_criterion(y_pred, y_target)

		if not self.latent_train and hasattr(self.qf, 'rew_classification'):
			num_pos = th.sum(terminals)
			num_neg = terminals.size(0) - num_pos
			rew_class, rew_class_kl_loss = self.qf.rew_classification(next_obs, eps=eps, return_kl=True,
																	  train_encoder=self.train_encoder_on_rew_class)
			rew_class_loss = th.nn.BCEWithLogitsLoss(pos_weight=num_neg / (num_pos + 1e-6))(rew_class, terminals)
			qf_loss += self.rew_class_weight * rew_class_loss + self.beta + rew_class_kl_loss

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
