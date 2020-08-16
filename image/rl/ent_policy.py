from railrl.torch.sac.awac_trainer import AWACTrainer
from railrl.torch.networks import ConcatMlp
import railrl.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
from railrl.core.logging import add_prefix
from railrl.torch.core import np_to_pytorch_batch
from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths
import torch
class PavlovTrainer(AWACTrainer):
	def __init__(self,env,policy,qf1,qf2,target_qf1,target_qf2,buffer_policy=None,
				penalty=-1,optimizer_class=optim.Adam,**kwargs):
		super().__init__(env,policy,qf1,qf2,target_qf1,target_qf2,buffer_policy,**kwargs)
		self.penalty = penalty

	def test_from_torch(self, batch):
		obs = batch['observations']
		actions = batch['actions']
		dist = self.policy(obs)
		prob = dist.log_prob(actions).reshape((-1,1)).exp()
		batch['rewards'] -= self.penalty*prob.detach()*batch['inputs']
		super().test_from_torch(batch)

	def train_from_torch(self, batch, train=True, pretrain=False,):
		obs = batch['observations']
		actions = batch['actions']
		dist = self.policy(obs)
		prob = dist.log_prob(actions).reshape((-1,1)).exp() # why are some log_probs positive?
		batch['rewards'] -= self.penalty*prob.detach()*batch['inputs']
		super().train_from_torch(batch,train=train,pretrain=pretrain)

class PavlovTrainer1(AWACTrainer):
	def __init__(self,env,policy,qf1,qf2,target_qf1,target_qf2,buffer_policy=None,
				penalty=1,af_lr=3e-4,optimizer_class=optim.Adam,**kwargs):
		super().__init__(env,policy,qf1,qf2,target_qf1,target_qf2,buffer_policy,**kwargs)
		self.penalty = penalty
		obs_dim = env.observation_space.low.size
		action_dim = env.action_space.low.size
		self.af1 = ConcatMlp(input_size=obs_dim + action_dim,output_size=2,hidden_sizes=[256]*3,)
		self.af2 = ConcatMlp(input_size=obs_dim + action_dim,output_size=2,hidden_sizes=[256]*3,)
		self.af_criterion = nn.CrossEntropyLoss()
		self.af1_optimizer = optimizer_class(self.af1.parameters(),lr=af_lr,)
		self.af2_optimizer = optimizer_class(self.af2.parameters(),lr=af_lr,)

	def test_from_torch(self, batch):
		obs = batch['observations']
		actions = batch['actions']
		inputs = batch['inputs'].flatten()

		a1_pred = self.af1(obs, actions)
		a2_pred = self.af2(obs, actions)
		af1_loss = self.af_criterion(a1_pred, inputs)
		af2_loss = self.af_criterion(a2_pred, inputs)

		self.eval_statistics['validation/AF1 Loss'] = np.mean(ptu.get_numpy(af1_loss))
		self.eval_statistics['validation/AF2 Loss'] = np.mean(ptu.get_numpy(af2_loss))
		super().test_from_torch(batch)

	def train_from_torch(self, batch, train=True, pretrain=False,):
		rewards = batch['rewards']
		terminals = batch['terminals']
		obs = batch['observations']
		actions = batch['actions']
		next_obs = batch['next_observations']
		inputs = batch['inputs'].flatten()
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
		# Make sure policy accounts for squashing functions like tanh correctly!
		next_dist = self.policy(next_obs)
		new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
		target_q_values = torch.min(
			self.target_qf1(next_obs, new_next_actions),
			self.target_qf2(next_obs, new_next_actions),
		) - alpha * new_log_pi.reshape((-1,1))

		q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
		qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
		qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

		"""
		AF Loss
		"""
		a1_pred = self.af1(obs, actions)
		a2_pred = self.af2(obs, actions)
		af1_loss = self.af_criterion(a1_pred, inputs.long())
		af2_loss = self.af_criterion(a2_pred, inputs.long())

		"""
		Policy Loss
		"""
		qf1_new_actions = self.qf1(obs, new_obs_actions)
		qf2_new_actions = self.qf2(obs, new_obs_actions)
		q_new_actions = torch.min(
			qf1_new_actions,
			qf2_new_actions,
		)
		af1_pred = self.af1(obs, actions)
		af2_pred = self.af2(obs, actions)

		# Advantage-weighted regression
		if self.awr_use_mle_for_vf:
			v1_pi = self.qf1(obs, policy_mle)
			v2_pi = self.qf2(obs, policy_mle)
			v_pi = torch.min(v1_pi, v2_pi)
		else:
			if self.vf_K > 1:
				vs = []
				for i in range(self.vf_K):
					u = dist.sample()
					q1 = self.qf1(obs, u)
					q2 = self.qf2(obs, u)
					v = torch.min(q1, q2)
					# v = q1
					vs.append(v)
				v_pi = torch.cat(vs, 1).mean(dim=1)
			else:
				# v_pi = self.qf1(obs, new_obs_actions)
				v1_pi = self.qf1(obs, new_obs_actions)
				v2_pi = self.qf2(obs, new_obs_actions)
				v_pi = torch.min(v1_pi, v2_pi)

		if self.awr_sample_actions:
			u = new_obs_actions
			if self.awr_min_q:
				q_adv = q_new_actions
			else:
				q_adv = qf1_new_actions
			af1_new_actions = self.af1(obs, new_obs_actions)
			af2_new_actions = self.af2(obs, new_obs_actions)
			af1_penalty = af1_new_actions.gather(1,torch.zeros((af1_new_actions.size()[0],1),dtype=int).cuda())
			af2_penalty = af2_new_actions.gather(1,torch.zeros((af2_new_actions.size()[0],1),dtype=int).cuda())
			input_adv = torch.min(af1_penalty,af2_penalty,)
		else:
			u = actions
			if self.awr_min_q:
				q_adv = torch.min(q1_pred, q2_pred)
			else:
				q_adv = q1_pred
			af1_penalty = af1_pred.gather(1,torch.zeros((af1_pred.size()[0],1),dtype=int).cuda())
			af2_penalty = af2_pred.gather(1,torch.zeros((af2_pred.size()[0],1),dtype=int).cuda())
			input_adv = torch.min(af1_penalty,af2_penalty,)

		policy_logpp = dist.log_prob(u)

		beta = self.beta_schedule.get_value(self._n_train_steps_total)

		if self.normalize_over_state == "advantage":
			score = q_adv - v_pi
			if self.mask_positive_advantage:
				score = torch.sign(score)
			score += self.penalty*input_adv.reshape((-1,1))
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

		"""
		Update networks
		"""
		if self._n_train_steps_total % self.policy_update_period == 0 and self.update_policy:
			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			self.policy_optimizer.step()

		if self._n_train_steps_total % self.q_update_period == 0:
			self.qf1_optimizer.zero_grad()
			qf1_loss.backward()
			self.qf1_optimizer.step()

			self.qf2_optimizer.zero_grad()
			qf2_loss.backward()
			self.qf2_optimizer.step()

		self.af1_optimizer.zero_grad()
		af1_loss.backward()
		self.af1_optimizer.step()

		self.af2_optimizer.zero_grad()
		af2_loss.backward()
		self.af2_optimizer.step()

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
			self.eval_statistics['AF1 Loss'] = np.mean(ptu.get_numpy(af1_loss))
			self.eval_statistics['AF2 Loss'] = np.mean(ptu.get_numpy(af2_loss))
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
				'A1 Predictions',
				ptu.get_numpy(a1_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'a2 Predictions',
				ptu.get_numpy(a2_pred),
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

	@property
	def networks(self):
		nets = [
			self.policy,
			self.qf1,
			self.qf2,
			self.target_qf1,
			self.target_qf2,
			self.af1,
			self.af2,
		]
		if self.buffer_policy:
			nets.append(self.buffer_policy)
		return nets

class PavlovTrainer2(AWACTrainer):
	def __init__(self,env,policy,qf1,qf2,target_qf1,target_qf2,buffer_policy=None,
				penalty=-1,optimizer_class=optim.Adam,**kwargs):
		super().__init__(env,policy,qf1,qf2,target_qf1,target_qf2,buffer_policy,**kwargs)
		self.penalty = penalty

	def train_from_torch(self, batch, train=True, pretrain=False,):
		rewards = batch['rewards']
		terminals = batch['terminals']
		obs = batch['observations']
		actions = batch['actions']
		next_obs = batch['next_observations']
		inputs = batch['inputs'].flatten()
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
		# Make sure policy accounts for squashing functions like tanh correctly!
		next_dist = self.policy(next_obs)
		new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
		target_q_values = torch.min(
			self.target_qf1(next_obs, new_next_actions),
			self.target_qf2(next_obs, new_next_actions),
		) - alpha * new_log_pi.reshape((-1,1))

		q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
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
		if self.awr_use_mle_for_vf:
			v1_pi = self.qf1(obs, policy_mle)
			v2_pi = self.qf2(obs, policy_mle)
			v_pi = torch.min(v1_pi, v2_pi)
		else:
			if self.vf_K > 1:
				vs = []
				for i in range(self.vf_K):
					u = dist.sample()
					q1 = self.qf1(obs, u)
					q2 = self.qf2(obs, u)
					v = torch.min(q1, q2)
					# v = q1
					vs.append(v)
				v_pi = torch.cat(vs, 1).mean(dim=1)
			else:
				# v_pi = self.qf1(obs, new_obs_actions)
				v1_pi = self.qf1(obs, new_obs_actions)
				v2_pi = self.qf2(obs, new_obs_actions)
				v_pi = torch.min(v1_pi, v2_pi)

		if self.awr_sample_actions:
			u = new_obs_actions
			if self.awr_min_q:
				q_adv = q_new_actions
			else:
				q_adv = qf1_new_actions
		else:
			u = actions
			if self.awr_min_q:
				q_adv = torch.min(q1_pred, q2_pred)
			else:
				q_adv = q1_pred
		
		policy_logpp = dist.log_prob(u)

		beta = self.beta_schedule.get_value(self._n_train_steps_total)

		if self.normalize_over_state == "advantage":
			score = q_adv - v_pi
			if self.mask_positive_advantage:
				score = torch.sign(score)
			log_prob = 1/dist.log_prob(actions).reshape((-1,1))
			print(log_prob)
			score += self.penalty*log_prob.detach()*batch['inputs']
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

		"""
		Update networks
		"""
		if self._n_train_steps_total % self.policy_update_period == 0 and self.update_policy:
			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			self.policy_optimizer.step()

		if self._n_train_steps_total % self.q_update_period == 0:
			self.qf1_optimizer.zero_grad()
			qf1_loss.backward()
			self.qf1_optimizer.step()

			self.qf2_optimizer.zero_grad()
			qf2_loss.backward()
			self.qf2_optimizer.step()

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

class PavlovTrainer3(AWACTrainer):
	def __init__(self,env,policy,qf1,qf2,target_qf1,target_qf2,buffer_policy=None,
				penalty=1,af_lr=3e-4,optimizer_class=optim.Adam,**kwargs):
		super().__init__(env,policy,qf1,qf2,target_qf1,target_qf2,buffer_policy,**kwargs)
		self.penalty = penalty
		obs_dim = env.observation_space.low.size
		action_dim = env.action_space.low.size
		self.pf = ConcatMlp(input_size=obs_dim + action_dim,output_size=1,hidden_sizes=[256]*3,)
		self.pf_criterion = nn.NLLLoss()
		self.pf_optimizer = optimizer_class(self.af1.parameters(),lr=af_lr,)

	def test_from_torch(self, batch):
		obs = batch['observations']
		actions = batch['actions']
		inputs = batch['inputs'].flatten()

		a1_pred = self.af1(obs, actions)
		a2_pred = self.af2(obs, actions)
		af1_loss = self.af_criterion(a1_pred, inputs)
		af2_loss = self.af_criterion(a2_pred, inputs)

		self.eval_statistics['validation/AF1 Loss'] = np.mean(ptu.get_numpy(af1_loss))
		self.eval_statistics['validation/AF2 Loss'] = np.mean(ptu.get_numpy(af2_loss))
		super().test_from_torch(batch)

	def train_from_torch(self, batch, train=True, pretrain=False,):
		rewards = batch['rewards']
		terminals = batch['terminals']
		obs = batch['observations']
		actions = batch['actions']
		next_obs = batch['next_observations']
		inputs = batch['inputs'].flatten()
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
		# Make sure policy accounts for squashing functions like tanh correctly!
		next_dist = self.policy(next_obs)
		new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
		target_q_values = torch.min(
			self.target_qf1(next_obs, new_next_actions),
			self.target_qf2(next_obs, new_next_actions),
		) - alpha * new_log_pi.reshape((-1,1))

		q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
		qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
		qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

		"""
		AF Loss
		"""
		input_pred = self.pf(obs, actions)
		input_pred = torch.cat((1-input_pred,input_pred),1)
		pf_loss = self.pf_criterion(input_pred, inputs.long())

		"""
		Policy Loss
		"""
		qf1_new_actions = self.qf1(obs, new_obs_actions)
		qf2_new_actions = self.qf2(obs, new_obs_actions)
		q_new_actions = torch.min(
			qf1_new_actions,
			qf2_new_actions,
		)
		input_pred = self.pf(obs, actions)

		# Advantage-weighted regression
		if self.awr_use_mle_for_vf:
			v1_pi = self.qf1(obs, policy_mle)
			v2_pi = self.qf2(obs, policy_mle)
			v_pi = torch.min(v1_pi, v2_pi)
		else:
			if self.vf_K > 1:
				vs = []
				for i in range(self.vf_K):
					u = dist.sample()
					q1 = self.qf1(obs, u)
					q2 = self.qf2(obs, u)
					v = torch.min(q1, q2)
					# v = q1
					vs.append(v)
				v_pi = torch.cat(vs, 1).mean(dim=1)
			else:
				# v_pi = self.qf1(obs, new_obs_actions)
				v1_pi = self.qf1(obs, new_obs_actions)
				v2_pi = self.qf2(obs, new_obs_actions)
				v_pi = torch.min(v1_pi, v2_pi)

		if self.awr_sample_actions:
			u = new_obs_actions
			if self.awr_min_q:
				q_adv = q_new_actions
			else:
				q_adv = qf1_new_actions
			af1_new_actions = self.af1(obs, new_obs_actions)
			af2_new_actions = self.af2(obs, new_obs_actions)
			af1_penalty = af1_new_actions.gather(1,torch.zeros((af1_new_actions.size()[0],1),dtype=int).cuda())
			af2_penalty = af2_new_actions.gather(1,torch.zeros((af2_new_actions.size()[0],1),dtype=int).cuda())
			input_adv = torch.min(af1_penalty,af2_penalty,)
		else:
			u = actions
			if self.awr_min_q:
				q_adv = torch.min(q1_pred, q2_pred)
			else:
				q_adv = q1_pred
			af1_penalty = af1_pred.gather(1,torch.zeros((af1_pred.size()[0],1),dtype=int).cuda())
			af2_penalty = af2_pred.gather(1,torch.zeros((af2_pred.size()[0],1),dtype=int).cuda())
			input_adv = torch.min(af1_penalty,af2_penalty,)

		policy_logpp = dist.log_prob(u)

		beta = self.beta_schedule.get_value(self._n_train_steps_total)

		if self.normalize_over_state == "advantage":
			score = q_adv - v_pi
			if self.mask_positive_advantage:
				score = torch.sign(score)
			score += self.penalty*input_adv.reshape((-1,1))
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

		"""
		Update networks
		"""
		if self._n_train_steps_total % self.policy_update_period == 0 and self.update_policy:
			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			self.policy_optimizer.step()

		if self._n_train_steps_total % self.q_update_period == 0:
			self.qf1_optimizer.zero_grad()
			qf1_loss.backward()
			self.qf1_optimizer.step()

			self.qf2_optimizer.zero_grad()
			qf2_loss.backward()
			self.qf2_optimizer.step()

		self.af1_optimizer.zero_grad()
		af1_loss.backward()
		self.af1_optimizer.step()

		self.af2_optimizer.zero_grad()
		af2_loss.backward()
		self.af2_optimizer.step()

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
			self.eval_statistics['AF1 Loss'] = np.mean(ptu.get_numpy(af1_loss))
			self.eval_statistics['AF2 Loss'] = np.mean(ptu.get_numpy(af2_loss))
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
				'A1 Predictions',
				ptu.get_numpy(a1_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'a2 Predictions',
				ptu.get_numpy(a2_pred),
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

	@property
	def networks(self):
		nets = [
			self.policy,
			self.qf1,
			self.qf2,
			self.target_qf1,
			self.target_qf2,
			self.af1,
			self.af2,
		]
		if self.buffer_policy:
			nets.append(self.buffer_policy)
		return nets

import numpy as np
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
class PavlovReplayBuffer(EnvReplayBuffer):
	def __init__(self,max_replay_buffer_size,env,env_info_sizes=None):
		super().__init__(max_replay_buffer_size,env,env_info_sizes)
		self._inputs = np.zeros((max_replay_buffer_size, 1))

	def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
		self._inputs[self._top] = not kwargs['env_info'].get('noop',True)
		super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)

	def random_batch(self, batch_size):
		indices = np.random.randint(0, self._size, batch_size)
		batch = dict(
			observations=self._observations[indices],
			actions=self._actions[indices],
			rewards=self._rewards[indices],
			terminals=self._terminals[indices],
			next_observations=self._next_obs[indices],
			inputs=self._inputs[indices],
		)
		for key in self._env_info_keys:
			assert key not in batch.keys()
			batch[key] = self._env_infos[key][indices]
		return batch