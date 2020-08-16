import gym
import torch
import numpy as np

from railrl.samplers.data_collector import MdpPathCollector
from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.torch.torch_rl_algorithm import (
	TorchBatchRLAlgorithm,
	TorchOnlineRLAlgorithm,
)

from railrl.torch.networks import Mlp
from torch.nn import LSTM as PytorchLSTM
from torch.nn import Linear
from railrl.torch.dqn.double_dqn import DoubleDQNTrainer

from railrl.demos.source.mdp_path_loader import MDPPathLoader

from railrl.exploration_strategies.base import \
	PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_and_epislon import GaussianAndEpislonStrategy

from railrl.core import logger
from railrl.torch.core import np_to_pytorch_batch
from railrl.core.logging import add_prefix
import railrl.torch.pytorch_util as ptu
from railrl.envs.make_env import make
from railrl.misc.eval_util import create_stats_ordered_dict

import time

from railrl.torch.core import PyTorchModule
import torch.nn.functional as F
from railrl.torch.distributions import OneHotCategorical as TorchOneHot
from railrl.torch.distributions import Distribution
import torch.optim as optim

from replay_buffers import *

class LSTM(PytorchLSTM):
	def __init__(self, input_size, output_size, **kwargs):
		hidden_size = kwargs.pop('hidden_size',256)
		init_w = kwargs.pop('init_w',3e-3)
		super().__init__(input_size, hidden_size, **kwargs)
		self.last_fc = Linear(hidden_size, output_size)
		self.last_fc.weight.data.uniform_(-init_w, init_w)
		self.last_fc.bias.data.fill_(0)

	def forward(self, inputs, hx=None):
		if hx is not None:
			out,out_hx = super().forward(inputs,hx)
		else:
			out,out_hx = super().forward(inputs)
		return self.last_fc(out),out_hx

class OneHotCategorical(Distribution,TorchOneHot):
	def rsample_and_logprob(self):
		s = self.sample()
		log_p = self.log_prob(s)
		return s, log_p		

class ArgmaxDiscretePolicy(PyTorchModule):
	def __init__(self, qf1, qf2, pf, obs_dim, penalty=.01, q_coeff=1):
		super().__init__()
		self.qf1 = qf1
		self.qf2 = qf2
		self.pf = pf
		self.penalty = penalty
		self.q_coeff = q_coeff
		self.obs_dim = obs_dim
		self.pf_hx = None

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = torch.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		obs, concat_obs = obs[:self.obs_dim], obs[self.obs_dim:]
		obs, concat_obs = obs[:self.obs_dim], obs[self.obs_dim:]
		with torch.no_grad():
			q_values = torch.min(self.qf1(concat_obs),self.qf2(concat_obs))
			input_prediction,self.pf_hx = self.pf(obs.reshape((1,1,-1)),self.pf_hx)
			logits = self.q_coeff*q_values + self.penalty*(1-torch.sigmoid(input_prediction.squeeze()))
			action = F.one_hot(logits.argmax(0,keepdim=True),list(logits.size())[0]).cpu().flatten().detach().numpy()
		return action, {}

	def reset(self):
		self.pf_hx = (torch.zeros((1,1,self.pf.hidden_size)),torch.zeros((1,1,self.pf.hidden_size)))

class EpsilonGreedyPolicy(PyTorchModule):
	def __init__(self, qf1, qf2, pf, num_actions, epsilon=.25, penalty=.01, q_coeff=1):
		super().__init__()
		self.qf1 = qf1
		self.qf2 = qf2
		self.pf = pf
		self.penalty = penalty
		self.q_coeff = q_coeff
		self.num_actions = num_actions
		self.epsilon = epsilon
		self.pf_hx = None

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = torch.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()
		input_prediction,self.pf_hx = self.pf(obs.unsqueeze(0),self.pf_hx)
		q_values = self.q_coeff*torch.min(self.qf1(obs),self.qf2(obs)) + self.penalty*(1-torch.sigmoid(input_prediction))
		index = q_values.argmax(0,keepdim=True) if np.random.random() > self.epsilon else torch.randint(0,self.num_actions,tuple(q_values.argmax(0,keepdim=True).size()))
		action = F.one_hot(index,list(q_values.size())[0]).cpu().flatten().detach().numpy()
		return action, {}

	def reset(self):
		self.pf_hx = (torch.zeros((1,1,self.pf.hidden_size)),torch.zeros((1,1,self.pf.hidden_size)))

class BoltzmannPolicy(PyTorchModule):
	def __init__(self, qf1, qf2, pf, obs_dim, penalty=.01, q_coeff=1, logit_scale=100):
		super().__init__()
		self.qf1 = qf1
		self.qf2 = qf2
		self.pf = pf

		self.penalty = penalty
		self.q_coeff = q_coeff
		self.logit_scale = 100
		self.obs_dim = obs_dim

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = torch.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		obs, concat_obs = obs[:self.obs_dim], obs[self.obs_dim:]
		with torch.no_grad():
			q_values = torch.min(self.qf1(concat_obs),self.qf2(concat_obs))
			input_prediction,self.pf_hx = self.pf(obs.reshape((1,1,-1)),self.pf_hx)

			logits = self.q_coeff*q_values + self.penalty*(1-torch.sigmoid(input_prediction.squeeze()))
			action = OneHotCategorical(logits=self.logit_scale*logits).sample().cpu().flatten().detach().numpy()
		return action, {}

	def reset(self):
		self.pf_hx = (torch.zeros((1,1,self.pf.hidden_size)),torch.zeros((1,1,self.pf.hidden_size)))

class DQNPavlovTrainer(DoubleDQNTrainer):
	def __init__(self,qf1,target_qf1,qf2,target_qf2,pf,obs_dim,pretrain_steps=int(5e4),**kwargs):
		super().__init__(qf1,target_qf1,**kwargs)
		self.obs_dim = obs_dim
		self.pretrain_steps = pretrain_steps
		self.qf1 = qf1
		self.target_qf1 = target_qf1
		self.qf2 = qf2
		self.target_qf2 = target_qf2
		self.qf1_optimizer = self.qf_optimizer
		self.qf2_optimizer = optim.Adam(
			self.qf2.parameters(),
			lr=self.learning_rate,
		)
		self.pf = pf
		self.pf_optimizer = optim.Adam(
			self.pf.parameters(),
			lr=self.learning_rate,
		)

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
		rewards = batch['rewards']
		terminals = batch['terminals']
		obs = batch['observations'][:,:,:self.obs_dim]
		actions = batch['actions']
		next_obs = batch['next_observations'][:,:,:self.obs_dim]
		inputs = batch['inputs']

		concat_obs = batch['observations'][:,:,self.obs_dim:]
		concat_next_obs = batch['next_observations'][:,:,self.obs_dim:]

		"""
		Compute loss
		"""

		best_action_idxs = torch.min(self.qf1(concat_next_obs),self.qf2(concat_next_obs)).max(
			1, keepdim=True
		)[1]
		target_q_values = torch.min(self.target_qf1(concat_next_obs).gather(
											1, best_action_idxs
										).detach(),
									self.target_qf2(concat_next_obs).gather(
											1, best_action_idxs
										).detach()
								)
		y_target = rewards + (1. - terminals) * self.discount * target_q_values
		y_target = y_target.detach()
		# actions is a one-hot vector
		y1_pred = torch.sum(self.qf1(concat_obs) * actions, dim=1, keepdim=True)
		qf1_loss = self.qf_criterion(y1_pred, y_target)
		y2_pred = torch.sum(self.qf2(concat_obs) * actions, dim=1, keepdim=True)
		qf2_loss = self.qf_criterion(y2_pred, y_target)

		# actions is a one-hot vector
		input_pred = torch.sigmoid(torch.sum(self.pf(obs.transpose(0,1)) * actions, dim=1, keepdim=True))
		input_pred = torch.cat((1.-input_pred,input_pred),1)
		pf_loss = -torch.gather(input_pred,1,inputs.transpose(0,1).long()).log().mean()

		"""
		Update networks
		"""
		self.qf1_optimizer.zero_grad()
		qf1_loss.backward()
		self.qf1_optimizer.step()
		self.qf2_optimizer.zero_grad()
		qf2_loss.backward()
		self.qf2_optimizer.step()

		self.pf_optimizer.zero_grad()
		pf_loss.backward()
		self.pf_optimizer.step()

		"""
		Soft target network updates
		"""
		if self._n_train_steps_total % self.target_update_period == 0:
			ptu.soft_update_from_to(
				self.qf1, self.target_qf1, self.soft_target_tau
			)
			ptu.soft_update_from_to(
				self.qf2, self.target_qf2, self.soft_target_tau
			)

		"""
		Save some statistics for eval using just one batch.
		"""
		if self._need_to_update_eval_statistics:
			self._need_to_update_eval_statistics = False
			self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
			self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
			self.eval_statistics['PF Loss'] = np.mean(ptu.get_numpy(pf_loss))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Y Predictions',
				ptu.get_numpy(torch.min(y1_pred,y2_pred)),
			))

	@property
	def networks(self):
		nets = [
			self.qf1,
			self.target_qf1,
			self.qf2,
			self.target_qf2,
			self.pf,
		]
		return nets

def experiment(variant):
	normalize_env = variant.get('normalize_env', True)
	env_id = variant.get('env_id', None)
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})

	expl_env = make(env_id, env_class, env_kwargs, normalize_env)
	eval_env = make(env_id, env_class, env_kwargs, normalize_env)
	expl_env.seed(variant['seedid'])
	eval_env.seed(variant['seedid'])

	if variant.get('add_env_demos', False):
		variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])
	if variant.get('add_env_offpolicy_data', False):
		variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

	path_loader_kwargs = variant.get("path_loader_kwargs", {})
	obs_dim = expl_env.observation_space.low.size
	action_dim = eval_env.action_space.low.size

	if hasattr(expl_env, 'info_sizes'):
		env_info_sizes = expl_env.info_sizes
	else:
		env_info_sizes = dict()

	qf_kwargs = variant.get("qf_kwargs", {})
	qf1 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		**qf_kwargs
	)
	target_qf1 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		**qf_kwargs
	)
	qf2 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		**qf_kwargs
	)
	target_qf2 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		**qf_kwargs
	)
	pf_kwargs = variant.get("pf_kwargs", {})
	pf = LSTM(
		input_size=obs_dim,
		output_size=action_dim,
		**pf_kwargs
	)
	expl_env.pf = pf
	eval_env.pf = pf

	policy_kwargs = variant['policy_kwargs']
	policy = ArgmaxDiscretePolicy(
		qf1=qf1,
		qf2=qf2,
		pf=pf,
		obs_dim=obs_dim,
		**policy_kwargs,
	)

	eval_policy = policy
	eval_path_collector = MdpPathCollector(
		eval_env,
		eval_policy,
	)

	expl_policy = policy
	exploration_kwargs =  variant.get('exploration_kwargs', {})
	if exploration_kwargs:
		exploration_strategy = exploration_kwargs.get("strategy", 'boltzmann')
		if exploration_strategy is None:
			pass
		elif exploration_strategy == 'boltzmann':
			expl_policy = BoltzmannPolicy(qf1,qf2,pf,obs_dim,logit_scale=exploration_kwargs['logit_scale'],**policy_kwargs)
		elif exploration_strategy == 'epsilon_greedy':
			expl_policy = EpsilonGreedyPolicy(qf1,qf2,pf,action_dim,epsilon=exploration_kwargs['epsilon'],**policy_kwargs)
		else:
			error

	replay_buffer_kwargs = variant.get('replay_buffer_kwargs',{})
	replay_buffer_kwargs['env'] = expl_env
	replay_buffer = variant.get('replay_buffer_class', PavlovSubtrajReplayBuffer)(
		**replay_buffer_kwargs,
	)
	
	trainer_class = variant.get("trainer_class", DQNPavlovTrainer)
	if trainer_class == DQNPavlovTrainer:
		trainer = trainer_class(
			qf1=qf1,
			target_qf1=target_qf1,
			qf2=qf2,
			target_qf2=target_qf2,
			pf=pf,
			obs_dim=obs_dim,
			**variant['trainer_kwargs']
		)
	else:
		error
	
	expl_path_collector = MdpPathCollector(
		expl_env,
		expl_policy,
	)
	algorithm = TorchBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=expl_env,
		evaluation_env=eval_env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		max_path_length=variant['max_path_length'],
		batch_size=variant['batch_size'],
		num_epochs=variant['num_epochs'],
		num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
		num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
		num_trains_per_train_loop=variant['num_trains_per_train_loop'],
		min_num_steps_before_training=variant['min_num_steps_before_training'],
	)
	algorithm.to(ptu.device)

	demo_train_buffer = PavlovSubtrajReplayBuffer(
		**replay_buffer_kwargs,
	)
	demo_test_buffer = PavlovSubtrajReplayBuffer(
		**replay_buffer_kwargs,
	)

	if variant.get('load_demos', False):
		path_loader_class = variant.get('path_loader_class', MDPPathLoader)
		path_loader = path_loader_class(trainer,
			replay_buffer=replay_buffer,
			demo_train_buffer=demo_train_buffer,
			demo_test_buffer=demo_test_buffer,
			**path_loader_kwargs
		)
		path_loader.load_demos()
		print(replay_buffer.num_steps_can_sample())
	if variant.get('pretrain_rl', False):
		trainer.pretrain()
	if variant.get('train_rl', True):
		# expl_env.render('human')

		algorithm.train()
		