import gym
import torch
import numpy as np
import os

from railrl.samplers.data_collector import MdpPathCollector
from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.torch.torch_rl_algorithm import (
	TorchBatchRLAlgorithm,
	TorchOnlineRLAlgorithm,
)

from railrl.torch.networks import Mlp
from torch.nn import LSTM as PytorchLSTM
from torch.nn import Linear, PReLU
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
from torch.nn import CrossEntropyLoss
from railrl.pythonplusplus import identity

from collections import OrderedDict
from railrl.core.timer import timer

from replay_buffers import *
from tqdm import tqdm,trange

class LSTM(PyTorchModule):
	def __init__(self, input_size, output_size, output_activation=identity, **kwargs):
		super().__init__()
		hidden_size = kwargs.pop('hidden_size',256)
		init_w = kwargs.pop('init_w',3e-3)
		self.lstm = PytorchLSTM(input_size, hidden_size, **kwargs)
		self.last_fc = Linear(hidden_size, output_size)
		# self.last_fc.weight.data.uniform_(-init_w, init_w)
		# self.last_fc.bias.data.fill_(0)
		self.output_activation = output_activation
		self.input_size = input_size
		self.hidden_size = self.lstm.hidden_size

	def forward(self, inputs, hx=None):
		if hx is not None:
			out,out_hx = self.lstm.forward(inputs,hx)
		else:
			out,out_hx = self.lstm.forward(inputs)
		return self.output_activation(self.last_fc(out)),out_hx

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
		self.action_dim = qf1.output_size

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = torch.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		obs, concat_obs = obs[:self.obs_dim], obs
		with torch.no_grad():
			# input_prediction, pf_hxs = torch.zeros(self.action_dim),[]
			input_predictions = []
			for i,oh_action in enumerate(F.one_hot(torch.arange(0,self.action_dim),self.action_dim)):
				if next(self.qf1.parameters()).is_cuda:
					oh_action = oh_action.cuda()
					input_prediction = input_prediction.cuda()
				single_prediction,pf_hx = self.pf(torch.cat((obs,oh_action)).reshape((1,1,-1)),self.pf_hx)
				# input_prediction[i] = single_prediction.squeeze().item()
				input_predictions.append(single_prediction.squeeze())
			input_prediction = torch.cat(input_predictions)

			# Uses t-1 hidden state with current input prediction
			concat_obs = torch.cat((concat_obs,self.pf_hx[0].squeeze(),self.pf_hx[1].squeeze(),input_prediction,))
			q_values = torch.min(self.qf1(concat_obs),self.qf2(concat_obs))

			action = F.one_hot(q_values.argmax(0,keepdim=True),self.action_dim).flatten().detach()
			_prediction, self.pf_hx = self.pf(torch.cat((obs,action)).reshape((1,1,-1)),self.pf_hx)
			self.prev_prediction = action
		return action.cpu().numpy(), {}

	def reset(self):
		if next(self.qf1.parameters()).is_cuda:
			self.pf_hx = (torch.zeros((1,1,self.pf.hidden_size)).cuda(),torch.zeros((1,1,self.pf.hidden_size)).cuda())
			self.prev_prediction = torch.zeros(self.action_dim).cuda()
		else:
			self.pf_hx = (torch.zeros((1,1,self.pf.hidden_size)),torch.zeros((1,1,self.pf.hidden_size)))
			self.prev_prediction = torch.zeros(self.action_dim)

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
		self.action_dim = qf1.output_size

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = torch.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		obs, concat_obs = obs[:self.obs_dim], obs
		with torch.no_grad():
			# input_prediction, pf_hxs = torch.zeros(self.action_dim),[]
			input_predictions = []
			for i,oh_action in enumerate(F.one_hot(torch.arange(0,self.action_dim),self.action_dim)):
				if next(self.qf1.parameters()).is_cuda:
					oh_action = oh_action.cuda()
					input_prediction = input_prediction.cuda()
				single_prediction,pf_hx = self.pf(torch.cat((obs,oh_action)).reshape((1,1,-1)),self.pf_hx)
				# input_prediction[i] = single_prediction.squeeze().item()
				input_predictions.append(single_prediction)
			input_prediction = torch.cat(input_predictions,dim=1)

			# Uses t-1 hidden state with current input prediction
			concat_obs = torch.cat((concat_obs,self.pf_hx[0].squeeze(),self.pf_hx[1].squeeze(),input_prediction,))
			q_values = torch.min(self.qf1(concat_obs),self.qf2(concat_obs))

			action = OneHotCategorical(logits=self.logit_scale*q_values).sample().flatten().detach()
			_prediction, self.pf_hx = self.pf(torch.cat((obs,action)).reshape((1,1,-1)),self.pf_hx)
			self.prev_prediction = action
		return action.cpu().numpy(), {}

	def reset(self):
		if next(self.qf1.parameters()).is_cuda:
			self.pf_hx = (torch.zeros((1,1,self.pf.hidden_size)).cuda(),torch.zeros((1,1,self.pf.hidden_size)).cuda())
			self.prev_prediction = torch.zeros(self.action_dim).cuda()
		else:
			self.pf_hx = (torch.zeros((1,1,self.pf.hidden_size)),torch.zeros((1,1,self.pf.hidden_size)))
			self.prev_prediction = torch.zeros(self.action_dim)

class PavlovBatchRLAlgorithm(TorchBatchRLAlgorithm):
	def __init__(self, num_pf_trains_per_train_loop, traj_batch_size, user_eval_mode=False, *args, **kwargs):
		super().__init__(*args,**kwargs)
		self.num_pf_trains_per_train_loop = num_pf_trains_per_train_loop
		self.traj_batch_size = traj_batch_size
		self.user_eval_mode = user_eval_mode

	def _train(self):
		done = (self.epoch == self.num_epochs)
		if done:
			return OrderedDict(), done

		if self.epoch == 0 and self.min_num_steps_before_training > 0:
			init_expl_paths = self.expl_data_collector.collect_new_paths(
				self.max_path_length,
				self.min_num_steps_before_training,
				discard_incomplete_paths=False,
			)
			self.replay_buffer.add_paths(init_expl_paths)
			self.expl_data_collector.end_epoch(-1)

		timer.start_timer('evaluation sampling')
		if not self.user_eval_mode and self.epoch % self._eval_epoch_freq == 0:
			self.eval_data_collector.collect_new_paths(
				self.max_path_length,
				self.num_eval_steps_per_epoch,
				discard_incomplete_paths=True,
			)
		timer.stop_timer('evaluation sampling')

		if not self._eval_only:
			for _ in range(self.num_train_loops_per_epoch):
				timer.start_timer('exploration sampling', unique=False)
				new_expl_paths = self.expl_data_collector.collect_new_paths(
					self.max_path_length,
					self.num_expl_steps_per_train_loop,
					discard_incomplete_paths=True,
				)
				timer.stop_timer('exploration sampling')

				timer.start_timer('replay buffer data storing', unique=False)
				self.replay_buffer.add_paths(new_expl_paths)
				timer.stop_timer('replay buffer data storing')

				timer.start_timer('training', unique=False)
				print(len(self.replay_buffer._change_start_indices))
				if not self.user_eval_mode or self.epoch == 0:
					self.trainer.reset_pf()
					for _ in range(self.num_pf_trains_per_train_loop):
						train_data = self.replay_buffer.random_traj_batch(self.traj_batch_size)
						self.trainer.train_pf(train_data)
					with torch.no_grad():
						self.replay_buffer.update_embeddings()
				for _ in range(self.num_trains_per_train_loop):
					train_data = self.replay_buffer.random_batch(self.batch_size)
					self.trainer.train(train_data)
				timer.stop_timer('training')
		log_stats = self._get_diagnostics()
		return log_stats, False

class DQNPavlovTrainer(DoubleDQNTrainer):
	def __init__(self,qf1,target_qf1,qf2,target_qf2,pf,obs_dim,pf_lr=3e-4,pf_decay=1e-4,**kwargs):
		super().__init__(qf1,target_qf1,**kwargs)
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
		self.pf_lr = pf_lr
		self.pf_optimizer = optim.Adam(
			self.pf.parameters(),
			weight_decay=pf_decay,
			lr=pf_lr,
		)
		self.obs_dim = obs_dim
		self.action_dim = qf1.output_size

	def reset_pf(self):
		for name, module in self.pf.named_children():
			module.reset_parameters()

	def train_pf(self, batch):
		batch = np_to_pytorch_batch(batch)

		rewards = batch['rewards'].transpose(0,1)
		terminals = batch['terminals'].transpose(0,1)
		obs = batch['observations'][:,:,:self.obs_dim].transpose(0,1)
		actions = batch['actions'].transpose(0,1)
		next_obs = batch['next_observations'][:,:,:self.obs_dim].transpose(0,1)
		# inputs = batch['inputs'].transpose(0,1)
		# targets = batch['targets'].transpose(0,1)
		recommends = batch['recommends'].transpose(0,1)


		"""
		Prediction loss
		"""
		# input_pred, _pf_hx = self.pf(torch.cat((obs,actions),dim=2))
		# input_pred = torch.cat((1.-input_pred,input_pred),dim=2)
		# # input_pred = torch.clamp(input_pred,min=1e-9,max=1-1e-9)
		# if next(self.qf1.parameters()).is_cuda:
		# 	weights = torch.tensor([.2,.8]).cuda()
		# else:
		# 	weights = torch.tensor([.2,.8])
		# pf_loss = -torch.gather(input_pred.log()*weights,2,inputs.long()).mean()*2
		# pf_accuracy = torch.eq(input_pred.clone().detach().max(2,keepdim=True)[1],inputs).float().mean()
		# pf_loss1 = -torch.gather(input_pred.clone().detach(),2,inputs.long()).log().mean()

		target_pred, _pf_hx = self.pf(torch.cat((obs,actions),dim=2))
		p = .2
		if next(self.qf1.parameters()).is_cuda:
			weights = torch.tensor([1,1,1,1,1,1,p/6/(1-p)]).cuda()
		else:
			weights = torch.tensor([1,1,1,1,1,1,p/6/(1-p)])
		pf_loss = CrossEntropyLoss(weight=weights)(target_pred.reshape((-1,7)),recommends.flatten().long())
		pf_accuracy = torch.eq(target_pred.clone().detach().max(2,keepdim=True)[1],recommends).float().mean()

		"""
		Update Prediction network
		"""
		self.pf_optimizer.zero_grad()
		pf_loss.backward()
		self.pf_optimizer.step()

		"""
		Save some statistics for eval using just one batch.
		"""
		# self.eval_statistics['PF Loss'] = np.mean(ptu.get_numpy(pf_loss1))
		self.eval_statistics['PF Weighted Loss'] = np.mean(ptu.get_numpy(pf_loss))
		self.eval_statistics['PF Accuracy'] = np.mean(ptu.get_numpy(pf_accuracy))

	def train_from_torch(self, batch):
		rewards = batch['rewards']
		terminals = batch['terminals']
		actions = batch['actions']
		concat_obs = batch['observations']
		concat_next_obs = batch['next_observations']

		"""
		Q loss
		"""
		best_action_idxs = torch.min(self.qf1(concat_next_obs),self.qf2(concat_next_obs)).max(
			1, keepdim=True
		)[1]
		target_q_values = torch.min(self.target_qf1(concat_next_obs).gather(
											1, best_action_idxs
										),
									self.target_qf2(concat_next_obs).gather(
											1, best_action_idxs
										)
								)
		y_target = rewards + (1. - terminals) * self.discount * target_q_values
		y_target = y_target.detach()
		# actions is a one-hot vector
		y1_pred = torch.sum(self.qf1(concat_obs) * actions, dim=1, keepdim=True)
		qf1_loss = self.qf_criterion(y1_pred, y_target)
		y2_pred = torch.sum(self.qf2(concat_obs) * actions, dim=1, keepdim=True)
		qf2_loss = self.qf_criterion(y2_pred, y_target)

		"""
		Update Q networks
		"""
		self.qf1_optimizer.zero_grad()
		qf1_loss.backward()
		self.qf1_optimizer.step()
		self.qf2_optimizer.zero_grad()
		qf2_loss.backward()
		self.qf2_optimizer.step()

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
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q Predictions',
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

	def get_snapshot(self):
		return dict(
			qf1 = self.qf1,
			target_qf1 = self.target_qf1,
			qf2 = self.qf2,
			target_qf2 = self.target_qf2,
			pf = self.pf,
		)

def eval_exp(variant):
	normalize_env = variant.get('normalize_env', True)
	env_id = variant.get('env_id', None)
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})
	eval_env = make(env_id, env_class, env_kwargs, normalize_env)
	eval_env.seed(variant['seedid'])
	current_obs_dim = eval_env.current_obs_dim

	file_name = variant['file_name']
	qf1 = torch.load(file_name,map_location=torch.device("cpu"))['trainer/qf1']
	qf2 = torch.load(file_name,map_location=torch.device("cpu"))['trainer/qf2']
	pf = torch.load(file_name,map_location=torch.device("cpu"))['trainer/pf']
	
	policy_kwargs = variant['policy_kwargs']
	policy = ArgmaxDiscretePolicy(
		qf1=qf1,
		qf2=qf2,
		pf=pf,
		obs_dim=current_obs_dim,
		**policy_kwargs,
	)

	eval_policy = policy
	eval_path_collector = MdpPathCollector(
		eval_env,
		eval_policy,
	)

	eval_env.render('human')
	eval_path_collector.collect_new_paths(
		200,
		200*10,
		discard_incomplete_paths=True,
	)

def resume_exp(variant):
	normalize_env = variant.get('normalize_env', True)
	env_id = variant.get('env_id', None)
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})

	expl_env = make(env_id, env_class, env_kwargs, normalize_env)
	eval_env = make(env_id, env_class, env_kwargs, normalize_env)
	expl_env.seed(variant['seedid'])
	eval_env.seed(variant['seedid'])

	path_loader_kwargs = variant.get("path_loader_kwargs", {})
	action_dim = eval_env.action_space.low.size
	current_obs_dim = expl_env.current_obs_dim

	file_name = os.path.join(variant['file_name'],'params.pkl')
	qf1 = torch.load(file_name,map_location=torch.device("cpu"))['trainer/qf1']
	target_qf1 = torch.load(file_name,map_location=torch.device("cpu"))['trainer/target_qf1']
	qf2 = torch.load(file_name,map_location=torch.device("cpu"))['trainer/qf2']
	target_qf2 = torch.load(file_name,map_location=torch.device("cpu"))['trainer/target_qf2']
	pf = torch.load(file_name,map_location=torch.device("cpu"))['trainer/pf']

	policy_kwargs = variant['policy_kwargs']
	policy = ArgmaxDiscretePolicy(
		qf1=qf1,
		qf2=qf2,
		pf=pf,
		obs_dim=current_obs_dim,
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
			expl_policy = BoltzmannPolicy(qf1,qf2,pf,current_obs_dim,logit_scale=exploration_kwargs['logit_scale'],**policy_kwargs)
		elif exploration_strategy == 'epsilon_greedy':
			expl_policy = EpsilonGreedyPolicy(qf1,qf2,pf,action_dim,epsilon=exploration_kwargs['epsilon'],**policy_kwargs)
		else:
			error

	

	trainer_class = variant.get("trainer_class", DQNPavlovTrainer)
	if trainer_class == DQNPavlovTrainer:
		trainer = trainer_class(
			qf1=qf1,
			target_qf1=target_qf1,
			qf2=qf2,
			target_qf2=target_qf2,
			pf=pf,
			obs_dim=current_obs_dim,
			**variant['trainer_kwargs']
		)
	else:
		error

	expl_path_collector = MdpPathCollector(
		expl_env,
		expl_policy,
	)
	algorithm = PavlovBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=expl_env,
		evaluation_env=eval_env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		max_path_length=variant['max_path_length'],
		batch_size=variant['batch_size'],
		traj_batch_size=variant['traj_batch_size'],
		num_epochs=variant['num_epochs'],
		num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
		num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
		num_pf_trains_per_train_loop=variant['num_pf_trains_per_train_loop'],
		num_trains_per_train_loop=variant['num_trains_per_train_loop'],
		min_num_steps_before_training=variant['min_num_steps_before_training'],
		user_eval_mode=variant['user_eval_mode']
	)
	algorithm.to(ptu.device)
	algorithm.train()

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
	action_dim = eval_env.action_space.low.size
	current_obs_dim = expl_env.current_obs_dim

	pf_kwargs = variant.get("pf_kwargs", {})
	pf = LSTM(
		input_size=current_obs_dim + action_dim,
		output_size=7,
		# output_activation=torch.sigmoid,
		**pf_kwargs
	)
	obs_dim = expl_env.observation_space.low.size
	obs_dim += 2*pf.hidden_size + 7*expl_env.env.num_targets
	qf_kwargs = variant.get("qf_kwargs", {})
	qf1 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_activation=PReLU(),
		**qf_kwargs
	)
	target_qf1 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_activation=PReLU(),
		**qf_kwargs
	)
	qf2 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_activation=PReLU(),
		**qf_kwargs
	)
	target_qf2 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_activation=PReLU(),
		**qf_kwargs
	)

	policy_kwargs = variant['policy_kwargs']
	policy = ArgmaxDiscretePolicy(
		qf1=qf1,
		qf2=qf2,
		pf=pf,
		obs_dim=current_obs_dim,
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
			expl_policy = BoltzmannPolicy(qf1,qf2,pf,current_obs_dim,logit_scale=exploration_kwargs['logit_scale'],**policy_kwargs)
		else:
			error

	replay_buffer_kwargs = variant.get('replay_buffer_kwargs',{})
	replay_buffer_kwargs['env'] = expl_env
	replay_buffer = variant.get('replay_buffer_class', PavlovSubtrajReplayBuffer)(
		pf=pf,
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
			obs_dim=current_obs_dim,
			**variant['trainer_kwargs']
		)
	else:
		error

	expl_path_collector = MdpPathCollector(
		expl_env,
		expl_policy,
	)
	algorithm = PavlovBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=expl_env,
		evaluation_env=eval_env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		max_path_length=variant['max_path_length'],
		batch_size=variant['batch_size'],
		traj_batch_size=variant['traj_batch_size'],
		num_epochs=variant['num_epochs'],
		num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
		num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
		num_pf_trains_per_train_loop=variant['num_pf_trains_per_train_loop'],
		num_trains_per_train_loop=variant['num_trains_per_train_loop'],
		min_num_steps_before_training=variant['min_num_steps_before_training'],
		user_eval_mode=variant['user_eval_mode']
	)
	algorithm.to(ptu.device)

	if variant.get('load_demos', False):
		path_loader_class = variant.get('path_loader_class', MDPPathLoader)
		path_loader = path_loader_class(trainer,
			replay_buffer=replay_buffer,
			demo_train_buffer=replay_buffer,
			demo_test_buffer=replay_buffer,
			**path_loader_kwargs
		)
		path_loader.load_demos()
	if variant.get('train_rl', True):
		algorithm.train()
