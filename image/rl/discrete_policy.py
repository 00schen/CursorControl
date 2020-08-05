from railrl.torch.distributions import OneHotCategorical as TorchOneHot
from railrl.torch.distributions import Distribution
from railrl.torch.networks import Mlp
from railrl.torch.sac.policies.base import TorchStochasticPolicy
import torch
import numpy as np
class OneHotCategorical(Distribution,TorchOneHot):
	def rsample_and_logprob(self):
		s = self.sample()
		log_p = self.log_prob(s)
		return s, log_p
class CategoricalPolicy(Mlp, TorchStochasticPolicy):
	def __init__(
			self,
			hidden_sizes,
			obs_dim,
			action_dim,
			init_w=1e-3,
			**kwargs
	):
		super().__init__(
			hidden_sizes,
			input_size=obs_dim,
			output_size=action_dim,
			init_w=init_w,
			**kwargs
		)
		last_hidden_size = obs_dim
		if len(hidden_sizes) > 0:
			last_hidden_size = hidden_sizes[-1]
		self.last_fc_logits = nn.Linear(last_hidden_size, action_dim)
		self.last_fc_logits.weight.data.uniform_(-init_w, init_w)
		self.last_fc_logits.bias.data.uniform_(-init_w, init_w)

	def forward(self, obs):
		h = obs
		for i, fc in enumerate(self.fcs):
			h = self.hidden_activation(fc(h))
		mean = self.last_fc(h)
		logits = self.last_fc_logits(h)

		return OneHotCategorical(logits=logits)

	def logprob(self, action, logits):
		cat = OneHotCategorical(logits=logits)
		log_prob = cat.log_prob(action,)
		log_prob = log_prob.sum(dim=1, keepdim=True)
		return log_prob

from railrl.torch.sac.awac_trainer import AWACTrainer
from railrl.torch.networks import ConcatMlp
import railrl.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
from railrl.core.logging import add_prefix
from railrl.torch.core import np_to_pytorch_batch
from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths
class DiscreteTrainer(AWACTrainer):
	def train_from_torch(self, batch, train=True, pretrain=False,):
		rewards = batch['rewards']
		terminals = batch['terminals']
		obs = batch['observations']
		actions = batch['actions']
		next_obs = batch['next_observations']
		weights = batch.get('weights', None)
		if self.reward_transform:
			rewards = self.reward_transform(rewards)

		if self.terminal_transform:
			terminals = self.terminal_transform(terminals)
		"""
		Policy and Alpha Loss
		"""
		dist = self.policy(obs)
		new_obs_actions, log_pi = dist.rsample_and_logprob()
		policy_mle = dist.mle_estimate()

		if self.brac:
			buf_dist = self.buffer_policy(obs)
			buf_log_pi = buf_dist.log_prob(actions)
			rewards = rewards + buf_log_pi

		if self.use_automatic_entropy_tuning:
			alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
			alpha = self.log_alpha.exp()
		else:
			alpha_loss = 0
			alpha = self.alpha

		"""
		QF Loss
		"""
		q1_pred = self.qf1(obs, actions)
		q2_pred = self.qf2(obs, actions)

		# Advantage-weighted regression
		action_dim = self.env.action_space.low.size
		qs = []
		for action1 in F.one_hot(torch.arange(0,action_dim),action_dim):
			action1 = action1.float().reshape((1,-1)).repeat((obs.shape[0],1)).cuda()
			qs.append(torch.min(self.target_qf1(obs,action1),self.target_qf2(obs,action1)).cpu().detach().numpy().tolist())
		if (torch.tensor(qs) > 10).sum().item() > 0:
			print("large q")
		if torch.isinf(torch.tensor(qs).exp()).sum().item() > 0:
			print("infinite q")
		target_v = torch.tensor(qs).logsumexp(dim=0).cuda()
		# v_pi = torch.tensor(qs).exp().sum(dim=0).log().cuda()
		if torch.isnan(target_v).sum().item() > 0:
			print("v error")

		q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_v
		qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
		qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

		"""
		Policy Loss
		"""
		qf1_new_actions = self.qf1(obs, new_obs_actions)
		qf2_new_actions = self.qf2(obs, new_obs_actions)
		q_new_actions = torch.min(
			qf1_new_actions,
			qf2_new_actions,
		)

		# Advantage-weighted regression
		action_dim = self.env.action_space.low.size
		qs = []
		for action1 in F.one_hot(torch.arange(0,action_dim),action_dim):
			action1 = action1.float().reshape((1,-1)).repeat((obs.shape[0],1)).cuda()
			qs.append(torch.min(self.qf1(obs,action1),self.qf2(obs,action1)).cpu().detach().numpy().tolist())
		if (torch.tensor(qs) > 10).sum().item() > 0:
			print("large q")
		if torch.isinf(torch.tensor(qs).exp()).sum().item() > 0:
			print("infinite q")
		v_pi = torch.tensor(qs).logsumexp(dim=0).cuda()
		# v_pi = torch.tensor(qs).exp().sum(dim=0).log().cuda()
		if torch.isnan(v_pi).sum().item() > 0:
			print("v error")

		if self.awr_sample_actions:
			u = new_obs_actions
			if self.awr_min_q:
				q_adv = q_new_actions
			else:
				q_adv = qf1_new_actions
		elif self.buffer_policy_sample_actions:
			buf_dist = self.buffer_policy(obs)
			u, _ = buf_dist.rsample_and_logprob()
			qf1_buffer_actions = self.qf1(obs, u)
			qf2_buffer_actions = self.qf2(obs, u)
			q_buffer_actions = torch.min(
				qf1_buffer_actions,
				qf2_buffer_actions,
			)
			if self.awr_min_q:
				q_adv = q_buffer_actions
			else:
				q_adv = qf1_buffer_actions
		else:
			u = actions
			if self.awr_min_q:
				q_adv = torch.min(q1_pred, q2_pred)
			else:
				q_adv = q1_pred

		policy_logpp = dist.log_prob(u)

		if self.use_automatic_beta_tuning:
			buffer_dist = self.buffer_policy(obs)
			beta = self.log_beta.exp()
			kldiv = torch.distributions.kl.kl_divergence(dist, buffer_dist)
			beta_loss = -1*(beta*(kldiv-self.beta_epsilon).detach()).mean()

			self.beta_optimizer.zero_grad()
			beta_loss.backward()
			self.beta_optimizer.step()
		else:
			beta = self.beta_schedule.get_value(self._n_train_steps_total)

		if self.normalize_over_state == "advantage":
			score = q_adv - v_pi
			if self.mask_positive_advantage:
				score = torch.sign(score)
		elif self.normalize_over_state == "Z":
			buffer_dist = self.buffer_policy(obs)
			K = self.Z_K
			buffer_obs = []
			buffer_actions = []
			log_bs = []
			log_pis = []
			for i in range(K):
				u = buffer_dist.sample()
				log_b = buffer_dist.log_prob(u)
				log_pi = dist.log_prob(u)
				buffer_obs.append(obs)
				buffer_actions.append(u)
				log_bs.append(log_b)
				log_pis.append(log_pi)
			buffer_obs = torch.cat(buffer_obs, 0)
			buffer_actions = torch.cat(buffer_actions, 0)
			p_buffer = torch.exp(torch.cat(log_bs, 0).sum(dim=1, ))
			log_pi = torch.cat(log_pis, 0)
			log_pi = log_pi.sum(dim=1, )
			q1_b = self.qf1(buffer_obs, buffer_actions)
			q2_b = self.qf2(buffer_obs, buffer_actions)
			q_b = torch.min(q1_b, q2_b)
			q_b = torch.reshape(q_b, (-1, K))
			adv_b = q_b - v_pi
			# if self._n_train_steps_total % 100 == 0:
			#     import ipdb; ipdb.set_trace()
			# Z = torch.exp(adv_b / beta).mean(dim=1, keepdim=True)
			# score = torch.exp((q_adv - v_pi) / beta) / Z
			# score = score / sum(score)
			logK = torch.log(ptu.tensor(float(K)))
			logZ = torch.logsumexp(adv_b/beta - logK, dim=1, keepdim=True)
			logS = (q_adv - v_pi)/beta - logZ
			# logZ = torch.logsumexp(q_b/beta - logK, dim=1, keepdim=True)
			# logS = q_adv/beta - logZ
			score = F.softmax(logS, dim=0) # score / sum(score)
		else:
			error

		if self.clip_score is not None:
			score = torch.clamp(score, max=self.clip_score)

		if self.weight_loss and weights is None:
			if self.normalize_over_batch == True:
				weights = F.softmax(score / beta, dim=0)
			elif self.normalize_over_batch == "whiten":
				adv_mean = torch.mean(score)
				adv_std = torch.std(score) + 1e-5
				normalized_score = (score - adv_mean) / adv_std
				weights = torch.exp(normalized_score / beta)
			elif self.normalize_over_batch == "exp":
				weights = torch.exp(score / beta)
			elif self.normalize_over_batch == "step_fn":
				weights = (score > 0).float()
			elif self.normalize_over_batch == False:
				weights = score
			else:
				error
		weights = weights[:, 0]

		policy_loss = alpha * log_pi.mean()

		if self.use_awr_update and self.weight_loss:
			policy_loss = policy_loss + self.awr_weight * (-policy_logpp * len(weights)*weights.detach()).mean()
		elif self.use_awr_update:
			policy_loss = policy_loss + self.awr_weight * (-policy_logpp).mean()

		if self.use_reparam_update:
			policy_loss = policy_loss + self.reparam_weight * (-q_new_actions).mean()

		policy_loss = self.rl_weight * policy_loss
		if self.compute_bc:
			train_policy_loss, train_logp_loss, train_mse_loss, _ = self.run_bc_batch(self.demo_train_buffer, self.policy)
			policy_loss = policy_loss + self.bc_weight * train_policy_loss



		if not pretrain and self.buffer_policy_reset_period > 0 and self._n_train_steps_total % self.buffer_policy_reset_period==0:
			del self.buffer_policy_optimizer
			self.buffer_policy_optimizer =  self.optimizer_class(
				self.buffer_policy.parameters(),
				weight_decay=self.policy_weight_decay,
				lr=self.policy_lr,
			)
			self.optimizers[self.buffer_policy] = self.buffer_policy_optimizer
			for i in range(self.num_buffer_policy_train_steps_on_reset):
				if self.train_bc_on_rl_buffer:
					if self.advantage_weighted_buffer_loss:
						buffer_dist = self.buffer_policy(obs)
						buffer_u = actions
						buffer_new_obs_actions, _ = buffer_dist.rsample_and_logprob()
						buffer_policy_logpp = buffer_dist.log_prob(buffer_u)
						buffer_policy_logpp = buffer_policy_logpp[:, None]

						buffer_q1_pred = self.qf1(obs, buffer_u)
						buffer_q2_pred = self.qf2(obs, buffer_u)
						buffer_q_adv = torch.min(buffer_q1_pred, buffer_q2_pred)

						buffer_v1_pi = self.qf1(obs, buffer_new_obs_actions)
						buffer_v2_pi = self.qf2(obs, buffer_new_obs_actions)
						buffer_v_pi = torch.min(buffer_v1_pi, buffer_v2_pi)

						buffer_score = buffer_q_adv - buffer_v_pi
						buffer_weights = F.softmax(buffer_score / beta, dim=0)
						buffer_policy_loss = self.awr_weight * (-buffer_policy_logpp * len(buffer_weights)*buffer_weights.detach()).mean()
					else:
						buffer_policy_loss, buffer_train_logp_loss, buffer_train_mse_loss, _ = self.run_bc_batch(
						self.replay_buffer.train_replay_buffer, self.buffer_policy)

					self.buffer_policy_optimizer.zero_grad()
					buffer_policy_loss.backward(retain_graph=True)
					self.buffer_policy_optimizer.step()

		if self.train_bc_on_rl_buffer:
			if self.advantage_weighted_buffer_loss:
				buffer_dist = self.buffer_policy(obs)
				buffer_u = actions
				buffer_new_obs_actions, _ = buffer_dist.rsample_and_logprob()
				buffer_policy_logpp = buffer_dist.log_prob(buffer_u)
				buffer_policy_logpp = buffer_policy_logpp[:, None]

				buffer_q1_pred = self.qf1(obs, buffer_u)
				buffer_q2_pred = self.qf2(obs, buffer_u)
				buffer_q_adv = torch.min(buffer_q1_pred, buffer_q2_pred)

				buffer_v1_pi = self.qf1(obs, buffer_new_obs_actions)
				buffer_v2_pi = self.qf2(obs, buffer_new_obs_actions)
				buffer_v_pi = torch.min(buffer_v1_pi, buffer_v2_pi)

				buffer_score = buffer_q_adv - buffer_v_pi
				buffer_weights = F.softmax(buffer_score / beta, dim=0)
				buffer_policy_loss = self.awr_weight * (-buffer_policy_logpp * len(buffer_weights)*buffer_weights.detach()).mean()
			else:
				buffer_policy_loss, buffer_train_logp_loss, buffer_train_mse_loss, _ = self.run_bc_batch(
					self.replay_buffer.train_replay_buffer, self.buffer_policy)



		"""
		Update networks
		"""
		if self._n_train_steps_total % self.q_update_period == 0:
			self.qf1_optimizer.zero_grad()
			qf1_loss.backward()
			self.qf1_optimizer.step()

			self.qf2_optimizer.zero_grad()
			qf2_loss.backward()
			self.qf2_optimizer.step()

		if self._n_train_steps_total % self.policy_update_period == 0 and self.update_policy:
			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			self.policy_optimizer.step()

		if self.train_bc_on_rl_buffer and self._n_train_steps_total % self.policy_update_period == 0 :
			self.buffer_policy_optimizer.zero_grad()
			buffer_policy_loss.backward()
			self.buffer_policy_optimizer.step()



		"""
		Soft Updates
		"""
		if self._n_train_steps_total % self.target_update_period == 0:
			ptu.soft_update_from_to(
				self.qf1, self.target_qf1, self.soft_target_tau
			)
			ptu.soft_update_from_to(
				self.qf2, self.target_qf2, self.soft_target_tau
			)

		"""
		Save some statistics for eval
		"""
		if self._need_to_update_eval_statistics:
			self._need_to_update_eval_statistics = False
			"""
			Eval should set this to None.
			This way, these statistics are only computed for one batch.
			"""
			policy_loss = (log_pi - q_new_actions).mean()

			self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
			self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
			self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
				policy_loss
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q1 Predictions',
				ptu.get_numpy(q1_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q2 Predictions',
				ptu.get_numpy(q2_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q Targets',
				ptu.get_numpy(q_target),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Log Pis',
				ptu.get_numpy(log_pi),
			))
			policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
			self.eval_statistics.update(policy_statistics)
			self.eval_statistics.update(create_stats_ordered_dict(
				'Advantage Weights',
				ptu.get_numpy(weights),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Advantage Score',
				ptu.get_numpy(score),
			))

			if self.normalize_over_state == "Z":
				self.eval_statistics.update(create_stats_ordered_dict(
					'logZ',
					ptu.get_numpy(logZ),
				))

			if self.use_automatic_entropy_tuning:
				self.eval_statistics['Alpha'] = alpha.item()
				self.eval_statistics['Alpha Loss'] = alpha_loss.item()

			if self.compute_bc:
				test_policy_loss, test_logp_loss, test_mse_loss, _ = self.run_bc_batch(self.demo_test_buffer, self.policy)
				self.eval_statistics.update({
					"bc/Train Logprob Loss": ptu.get_numpy(train_logp_loss),
					"bc/Test Logprob Loss": ptu.get_numpy(test_logp_loss),
					"bc/Train MSE": ptu.get_numpy(train_mse_loss),
					"bc/Test MSE": ptu.get_numpy(test_mse_loss),
					"bc/train_policy_loss": ptu.get_numpy(train_policy_loss),
					"bc/test_policy_loss": ptu.get_numpy(test_policy_loss),
				})
			if self.train_bc_on_rl_buffer:
				_, buffer_train_logp_loss, _, _ = self.run_bc_batch(
					self.replay_buffer.train_replay_buffer,
					self.buffer_policy)

				_, buffer_test_logp_loss, _, _ = self.run_bc_batch(
					self.replay_buffer.validation_replay_buffer,
					self.buffer_policy)
				buffer_dist = self.buffer_policy(obs)
				kldiv = torch.distributions.kl.kl_divergence(dist, buffer_dist)

				_, train_offline_logp_loss, _, _ = self.run_bc_batch(
					self.demo_train_buffer,
					self.buffer_policy)

				_, test_offline_logp_loss, _, _ = self.run_bc_batch(
					self.demo_test_buffer,
					self.buffer_policy)

				self.eval_statistics.update({

					"buffer_policy/Train Online Logprob": -1 * ptu.get_numpy(buffer_train_logp_loss),
					"buffer_policy/Test Online Logprob": -1 * ptu.get_numpy(buffer_test_logp_loss),

					"buffer_policy/Train Offline Logprob": -1 * ptu.get_numpy(train_offline_logp_loss),
					"buffer_policy/Test Offline Logprob": -1 * ptu.get_numpy(test_offline_logp_loss),

					"buffer_policy/train_policy_loss": ptu.get_numpy(buffer_policy_loss),
					# "buffer_policy/test_policy_loss": ptu.get_numpy(buffer_test_policy_loss),
					"buffer_policy/kl_div": ptu.get_numpy(kldiv.mean()),
				})
			if self.use_automatic_beta_tuning:
				self.eval_statistics.update({
					"adaptive_beta/beta":ptu.get_numpy(beta.mean()),
					"adaptive_beta/beta loss": ptu.get_numpy(beta_loss.mean()),
				})

			if self.validation_qlearning:
				train_data = self.replay_buffer.validation_replay_buffer.random_batch(self.bc_batch_size)
				train_data = np_to_pytorch_batch(train_data)
				obs = train_data['observations']
				next_obs = train_data['next_observations']
				# goals = train_data['resampled_goals']
				train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
				train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
				self.test_from_torch(train_data)

		self._n_train_steps_total += 1
