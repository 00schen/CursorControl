import torch as th
import numpy as np
import os

from railrl.samplers.data_collector import MdpPathCollector
from railrl.torch.torch_rl_algorithm import (
	TorchBatchRLAlgorithm,
)

from railrl.torch.networks import Mlp
from railrl.torch.dqn.double_dqn import DoubleDQNTrainer
from torch.nn import LSTM as PytorchLSTM
from torch.nn import Linear, PReLU

from railrl.demos.source.mdp_path_loader import MDPPathLoader

from railrl.core import logger
from railrl.torch.core import np_to_pytorch_batch
import railrl.torch.pytorch_util as ptu
from railrl.envs.make_env import make
from railrl.misc.eval_util import create_stats_ordered_dict

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

class LSTM(PyTorchModule):
	def __init__(self, input_size, output_size, output_activation=identity, **kwargs):
		super().__init__()
		hidden_size = kwargs.pop('hidden_size',256)
		self.embed_fc1 = Linear(input_size,hidden_size)
		self.embed_fc2 = Linear(hidden_size,hidden_size)
		self.embed_fc3 = Linear(hidden_size,hidden_size)
		self.lstm = PytorchLSTM(hidden_size, hidden_size, **kwargs)
		self.last_fc = Linear(hidden_size, output_size)
		self.output_activation = output_activation
		self.input_size = input_size
		self.hidden_size = self.lstm.hidden_size
		self.output_size = output_size

	def forward(self, inputs, hx=None):
		embedding = F.leaky_relu(self.embed_fc1(inputs))
		embedding = F.leaky_relu(self.embed_fc2(embedding))
		embedding = F.leaky_relu(self.embed_fc3(embedding))

		if hx is not None:
			out,out_hx = self.lstm.forward(embedding,hx)
		else:
			out,out_hx = self.lstm.forward(embedding)
		return self.output_activation(self.last_fc(out)),out_hx

class OneHotCategorical(Distribution,TorchOneHot):
	def rsample_and_logprob(self):
		s = self.sample()
		log_p = self.log_prob(s)
		return s, log_p

class ArgmaxDiscretePolicy(PyTorchModule):
	def __init__(self, qf1, qf2, pf, obs_dim, env):
		super().__init__()
		self.qf1 = qf1
		self.qf2 = qf2
		self.pf = pf
		self.obs_dim = obs_dim
		self.action_dim = qf1.output_size
		self.env = env

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		obs, concat_obs = obs[-self.obs_dim:], obs[3:]
		# obs, concat_obs = obs[-self.obs_dim:], obs
		with th.no_grad():
			input_predictions = []
			for i,oh_action in enumerate(F.one_hot(th.arange(0,self.action_dim),self.action_dim)):
				if next(self.qf1.parameters()).is_cuda:
					oh_action = oh_action.cuda()
				single_prediction,pf_hx = self.pf(th.cat((obs,oh_action)).reshape((1,1,-1)),self.pf_hx)
				input_predictions.append(single_prediction.squeeze())
			input_prediction = th.mean(th.stack(input_predictions),dim=0)
			if next(self.qf1.parameters()).is_cuda:
				input_prediction = input_prediction.cuda()

			# Uses t-1 hidden state with current input prediction
			# concat_obs = th.cat((concat_obs,self.pf_hx[0].squeeze(),self.pf_hx[1].squeeze(),input_prediction,))
			concat_obs = th.cat((input_prediction,concat_obs,)).float()
			q_values = th.min(self.qf1(concat_obs),self.qf2(concat_obs))

			action = F.one_hot(q_values.argmax(0,keepdim=True),self.action_dim).flatten().detach()
			prediction, self.pf_hx = self.pf(th.cat((obs,action)).reshape((1,1,-1)),self.pf_hx)
		return action.cpu().numpy(), {'prediction': prediction.squeeze().cpu().detach().numpy(), }
		# return action.cpu().numpy(), {}

	def reset(self):
		if next(self.qf1.parameters()).is_cuda:
			self.pf_hx = (th.zeros((1,1,self.pf.hidden_size)).cuda(),th.zeros((1,1,self.pf.hidden_size)).cuda())
		else:
			self.pf_hx = (th.zeros((1,1,self.pf.hidden_size)),th.zeros((1,1,self.pf.hidden_size)))

class BoltzmannPolicy(PyTorchModule):
	def __init__(self, qf1, qf2, pf, obs_dim, env, logit_scale=100):
		super().__init__()
		self.qf1 = qf1
		self.qf2 = qf2
		self.pf = pf

		self.logit_scale = 100
		self.obs_dim = obs_dim
		self.action_dim = qf1.output_size
		self.env = env

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		obs, concat_obs = obs[-self.obs_dim:], obs[3:]
		# obs, concat_obs = obs[-self.obs_dim:], obs
		with th.no_grad():
			input_predictions = []
			for i,oh_action in enumerate(F.one_hot(th.arange(0,self.action_dim),self.action_dim)):
				if next(self.qf1.parameters()).is_cuda:
					oh_action = oh_action.cuda()
				single_prediction,pf_hx = self.pf(th.cat((obs,oh_action)).reshape((1,1,-1)),self.pf_hx)
				input_predictions.append(single_prediction.squeeze())
			input_prediction = th.mean(th.stack(input_predictions),dim=0)
			if next(self.qf1.parameters()).is_cuda:
				input_prediction = input_prediction.cuda()

			# Uses t-1 hidden state with current input prediction
			# concat_obs = th.cat((concat_obs,self.pf_hx[0].squeeze(),self.pf_hx[1].squeeze(),input_prediction,))
			concat_obs = th.cat((input_prediction,concat_obs,)).float()
			q_values = th.min(self.qf1(concat_obs),self.qf2(concat_obs))

			action = OneHotCategorical(logits=self.logit_scale*q_values).sample().flatten().detach()
			_pred, self.pf_hx = self.pf(th.cat((obs,action)).reshape((1,1,-1)),self.pf_hx)
		return action.cpu().numpy(), {}

	def reset(self):
		if next(self.qf1.parameters()).is_cuda:
			self.pf_hx = (th.zeros((1,1,self.pf.hidden_size)).cuda(),th.zeros((1,1,self.pf.hidden_size)).cuda())
		else:
			self.pf_hx = (th.zeros((1,1,self.pf.hidden_size)),th.zeros((1,1,self.pf.hidden_size)))

class HybridAgent:
	def __init__(self, policy):
		self.policy = policy
	def get_action(self,obs):
		recommend = obs[-6:]
		action,info = self.policy.get_action(obs)
		if np.count_nonzero(recommend):
			return recommend,info
		else:
			return action, info
	def reset(self):
		self.policy.reset()

class PavlovBatchRLAlgorithm(TorchBatchRLAlgorithm):
	def __init__(self, num_pf_trains_per_train_loop, pf_train_frequency, traj_batch_size, *args, **kwargs):
		super().__init__(*args,**kwargs)
		self.num_pf_trains_per_train_loop = num_pf_trains_per_train_loop
		self.pf_train_frequency = pf_train_frequency
		self.traj_batch_size = traj_batch_size

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
		if self.epoch % self._eval_epoch_freq == 0:
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
				if self.epoch % self.pf_train_frequency == 0:
					self.trainer.reset_pf()
					for _ in range(self.num_pf_trains_per_train_loop):
						train_data = self.replay_buffer.random_traj_batch(self.traj_batch_size)
						self.trainer.train_pf(train_data)
					# with th.no_grad():
					# 	self.replay_buffer.update_embeddings()
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
		obs = batch['observations'][:,:,-self.obs_dim:].transpose(0,1)
		actions = batch['actions'].transpose(0,1)
		next_obs = batch['next_observations'][:,:,-self.obs_dim:].transpose(0,1)
		targets = batch['targets'].transpose(0,1)
		lengths = batch['lengths']

		"""
		Prediction loss
		"""
		# input_pred, _pf_hx = self.pf(th.cat((obs,actions),dim=2))
		# input_pred = th.cat((1.-input_pred,input_pred),dim=2)
		# # input_pred = th.clamp(input_pred,min=1e-9,max=1-1e-9)
		# if next(self.qf1.parameters()).is_cuda:
		# 	weights = th.tensor([.2,.8]).cuda()
		# else:
		# 	weights = th.tensor([.2,.8])
		# pf_loss = -th.gather(input_pred.log()*weights,2,inputs.long()).mean()*2
		# pf_accuracy = th.eq(input_pred.clone().detach().max(2,keepdim=True)[1],inputs).float().mean()
		# pf_loss1 = -th.gather(input_pred.clone().detach(),2,inputs.long()).log().mean()

		target_pred, _pf_hx = self.pf(th.cat((obs,actions,),dim=2))
		pf_loss_grid = (target_pred-targets).pow(2).sum(dim=-1)
		# pf_loss_grid = CrossEntropyLoss(reduction='none')\
		# 	(target_pred.reshape((-1,list(target_pred.size())[-1])),targets.flatten().long()).reshape(list(target_pred.size())[:-1])
		# pf_accuracy = th.eq(target_pred.clone().detach().max(2,keepdim=True)[1],targets).float().squeeze()

		row_vector = th.arange(0, obs.size()[0], 1)
		if next(self.qf1.parameters()).is_cuda:
			row_vector = row_vector.cuda()
		mask = (row_vector < lengths).transpose(0,1)
		pf_loss = pf_loss_grid.masked_select(mask).mean()
		# pf_accuracy = pf_accuracy.masked_select(mask).mean()
		pf_accuracy = pf_loss_grid.masked_select(mask).pow(.5).mean()

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
		self.eval_statistics['Training PF Loss'] = np.mean(ptu.get_numpy(pf_loss))
		self.eval_statistics['PF Accuracy'] = np.mean(ptu.get_numpy(pf_accuracy))

	def val_pf(self, batch):
		batch = np_to_pytorch_batch(batch)

		rewards = batch['rewards'].transpose(0,1)
		terminals = batch['terminals'].transpose(0,1)
		obs = batch['observations'][:,:,-self.obs_dim:].transpose(0,1)
		actions = batch['actions'].transpose(0,1)
		next_obs = batch['next_observations'][:,:,-self.obs_dim:].transpose(0,1)
		targets = batch['targets'].transpose(0,1)
		lengths = batch['lengths']

		"""
		Prediction loss
		"""
		# input_pred, _pf_hx = self.pf(th.cat((obs,actions),dim=2))
		# input_pred = th.cat((1.-input_pred,input_pred),dim=2)
		# # input_pred = th.clamp(input_pred,min=1e-9,max=1-1e-9)
		# if next(self.qf1.parameters()).is_cuda:
		# 	weights = th.tensor([.2,.8]).cuda()
		# else:
		# 	weights = th.tensor([.2,.8])
		# pf_loss = -th.gather(input_pred.log()*weights,2,inputs.long()).mean()*2
		# pf_accuracy = th.eq(input_pred.clone().detach().max(2,keepdim=True)[1],inputs).float().mean()
		# pf_loss1 = -th.gather(input_pred.clone().detach(),2,inputs.long()).log().mean()

		with th.no_grad():
			target_pred, _pf_hx = self.pf(th.cat((obs,actions,),dim=2))
			pf_loss_grid = (target_pred-targets).pow(2).sum(dim=-1)
			# pf_loss_grid = CrossEntropyLoss(reduction='none')\
			# 	(target_pred.reshape((-1,list(target_pred.size())[-1])),targets.flatten().long()).reshape(list(target_pred.size())[:-1])
			# pf_accuracy = th.eq(target_pred.clone().detach().max(2,keepdim=True)[1],targets).float().squeeze()

			row_vector = th.arange(0, obs.size()[0], 1)
			if next(self.qf1.parameters()).is_cuda:
				row_vector = row_vector.cuda()
			mask = (row_vector < lengths).transpose(0,1)
			pf_loss = pf_loss_grid.masked_select(mask).mean()
			# pf_accuracy_mean = pf_accuracy.masked_select(mask).mean()
			# pf_accuracy_std = pf_accuracy.masked_select(mask).std()
			pf_accuracy_mean = pf_loss_grid.masked_select(mask).pow(.5).mean()
			pf_accuracy_std = pf_loss_grid.masked_select(mask).pow(.5).std()


		"""
		Save some statistics for eval using just one batch.
		"""
		# self.eval_statistics['PF Loss'] = np.mean(ptu.get_numpy(pf_loss1))
		self.eval_statistics['Validation PF Loss'] = np.mean(ptu.get_numpy(pf_loss))
		self.eval_statistics['Validation PF Accuracy Mean'] = np.mean(ptu.get_numpy(pf_accuracy_mean))
		self.eval_statistics['Validation PF Accuracy Std'] = np.mean(ptu.get_numpy(pf_accuracy_std))

	def train_from_torch(self, batch):
		rewards = batch['rewards']
		# terminals = batch['terminals']
		successes = batch['successes']
		actions = batch['actions']
		concat_obs = batch['observations']
		concat_next_obs = batch['next_observations']

		"""
		Q loss
		"""
		best_action_idxs = th.min(self.qf1(concat_next_obs),self.qf2(concat_next_obs)).max(
			1, keepdim=True
		)[1]
		target_q_values = th.min(self.target_qf1(concat_next_obs).gather(
											1, best_action_idxs
										),
									self.target_qf2(concat_next_obs).gather(
											1, best_action_idxs
										)
								)
		y_target = rewards + (1. - successes) * self.discount * target_q_values
		y_target = y_target.detach()
		# actions is a one-hot vector
		y1_pred = th.sum(self.qf1(concat_obs) * actions, dim=1, keepdim=True)
		qf1_loss = self.qf_criterion(y1_pred, y_target)
		y2_pred = th.sum(self.qf2(concat_obs) * actions, dim=1, keepdim=True)
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
				ptu.get_numpy(th.min(y1_pred,y2_pred)),
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

from types import MethodType
def demonstration_factory(base):
	class DemonstrationEnv(base):
		def __init__(self,config):
			super().__init__(config)
			self.target_index = -1

		def __iter__(self):
			return self
		def __next__(self):
			self.target_index += 1
			if self.target_index == self.env.num_targets:
				self.target_index = -1
				raise StopIteration
			return self

		def reset(self):
			target_index = self.target_index
			def generate_target(self,index):
				nonlocal target_index
				self.__class__.generate_target(self,target_index)
			self.env.generate_target = MethodType(generate_target,self.env)
			return super().reset()
	return DemonstrationEnv
from agents import DemonstrationAgent
def collect_demonstrations(variant):
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})
	env_class = demonstration_factory(env_class)
	env = make(None, env_class, env_kwargs, False)
	env.seed(variant['seedid'])

	path_collector = MdpPathCollector(
		env,
		DemonstrationAgent(env,lower_p=.2),
	)

	if variant.get('render',False):
		env.render('human')
	demo_kwargs = variant.get('demo_kwargs')
	paths = []
	for env_i in env:
		print(env.target_index)
		fail_paths = deque([],demo_kwargs['fails_per_success']*demo_kwargs['paths_per_target'])
		success_paths = deque([],demo_kwargs['paths_per_target'])
		while len(fail_paths) < fail_paths.maxlen or len(success_paths) < success_paths.maxlen:
			collected_paths = path_collector.collect_new_paths(
				demo_kwargs['path_length'],
				demo_kwargs['path_length']*demo_kwargs['paths_per_target'],
				discard_incomplete_paths=True,
			)
			for path in collected_paths:
				if path['env_infos'][-1]['task_success']:
					success_paths.append(path)
				else:
					fail_paths.append(path)
			print(len(fail_paths),len(success_paths))
		paths.extend(fail_paths)
		paths.extend(success_paths)

	return paths

def eval_exp(variant):
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})
	eval_env = make(None, env_class, env_kwargs, False)
	eval_env.seed(variant['seedid'])
	current_obs_dim = eval_env.current_obs_dim

	file_name = os.path.join(variant['save_path'],'params.pkl')
	qf1 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf1']
	qf2 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf2']
	pf = th.load(file_name,map_location=th.device("cpu"))['trainer/pf']

	policy_kwargs = variant['policy_kwargs']
	policy = ArgmaxDiscretePolicy(
		qf1=qf1,
		qf2=qf2,
		pf=pf,
		obs_dim=current_obs_dim,
		env=eval_env,
		**policy_kwargs,
	)

	eval_policy = policy
	eval_path_collector = MdpPathCollector(
		eval_env,
		eval_policy,
	)

	if variant.get('render',False):
		eval_env.render('human')
	eval_collected_paths = eval_path_collector.collect_new_paths(
		variant['algorithm_args']['max_path_length'],
		variant['algorithm_args']['num_eval_steps_per_epoch'],
		discard_incomplete_paths=True,
	)

	np.save(os.path.join(variant['save_path'],'evaluated_paths'), eval_collected_paths)

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
	qf1 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf1']
	target_qf1 = th.load(file_name,map_location=th.device("cpu"))['trainer/target_qf1']
	qf2 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf2']
	target_qf2 = th.load(file_name,map_location=th.device("cpu"))['trainer/target_qf2']
	pf = th.load(file_name,map_location=th.device("cpu"))['trainer/pf']

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

from railrl.core import logger
from tqdm import trange
def pf_exp(variant):
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})
	eval_env = make(None, env_class, env_kwargs, False)
	expl_env = make(None, env_class, env_kwargs, False)
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
		input_size=current_obs_dim+action_dim,
		# output_size=eval_env.env.num_targets,
		output_size=3,
		# output_activation=th.sigmoid,
		**pf_kwargs
	)
	_qf = Mlp(
		input_size=1,
		output_size=1,
		hidden_sizes=[1]
	)

	replay_buffer_kwargs = variant.get('replay_buffer_kwargs',{})
	replay_buffer_kwargs['env'] = expl_env
	replay_buffer = variant.get('replay_buffer_class', PavlovSubtrajReplayBuffer)(
		obs_dim=current_obs_dim,
		pf=pf,
		**replay_buffer_kwargs,
	)

	trainer_class = variant.get("trainer_class", DQNPavlovTrainer)
	if trainer_class == DQNPavlovTrainer:
		trainer = trainer_class(
			qf1=_qf,
			target_qf1=_qf,
			qf2=_qf,
			target_qf2=_qf,
			pf=pf,
			obs_dim=current_obs_dim,
			**variant['trainer_kwargs']
		)
	else:
		error

	path_loader_class = variant.get('path_loader_class', MDPPathLoader)
	path_loader = path_loader_class(trainer,
		replay_buffer=replay_buffer,
		demo_train_buffer=replay_buffer,
		demo_test_buffer=replay_buffer,
		**path_loader_kwargs
	)
	path_loader.load_demos()

	for net in trainer.networks:
		net.to(ptu.device)
	# for _ in range(20):
	for _ in range(1):
		for _i in range(variant['algorithm_args']['num_pf_trains_per_train_loop']):
			train_data = replay_buffer.random_traj_batch(variant['algorithm_args']['traj_batch_size'])
			trainer.train_pf(train_data)
			val_data = replay_buffer.val_random_traj_batch(variant['algorithm_args']['traj_batch_size'])
			trainer.val_pf(val_data)
		logger.record_dict(trainer.get_diagnostics())
		logger.dump_tabular(with_prefix=True, with_timestamp=False)

	predictions = [pf(th.cat((obs[:length[0],np.newaxis,-current_obs_dim:].float(),
								action[:length[0],np.newaxis,:].float()),dim=2))[0].squeeze().cpu().detach().numpy()
	 for obs,action,length in zip(th.tensor(replay_buffer._observations[:replay_buffer._top]).cuda(),
	 						th.tensor(replay_buffer._actions[:replay_buffer._top]).cuda(),
							replay_buffer._lengths[:replay_buffer._top])]
	np.save(os.path.join(variant['save_path'],'predictions_1000'), predictions)

def experiment(variant):
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})
	eval_env = make(None, env_class, env_kwargs, False)
	expl_env = make(None, env_class, env_kwargs, False)
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
		input_size=current_obs_dim+action_dim,
		# output_size=expl_env.env.num_targets,
		output_size=3,
		# output_activation=th.sigmoid,
		**pf_kwargs
	)
	obs_dim = expl_env.observation_space.low.size
	# obs_dim += 2*pf.hidden_size + expl_env.env.num_targets
	# obs_dim += expl_env.env.num_targets
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
		env=eval_env,
		**policy_kwargs,
	)

	eval_policy = HybridAgent(policy)
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
			expl_policy = BoltzmannPolicy(qf1,qf2,pf,current_obs_dim,
										logit_scale=exploration_kwargs['logit_scale'],
										env=expl_env,
										**policy_kwargs)
			expl_policy = HybridAgent(expl_policy)
		else:
			error

	replay_buffer_kwargs = variant.get('replay_buffer_kwargs',{})
	replay_buffer_kwargs['env'] = expl_env
	replay_buffer = variant.get('replay_buffer_class', PavlovSubtrajReplayBuffer)(
		obs_dim=obs_dim,
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
		**variant['algorithm_args']
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
		print(replay_buffer._top)
		algorithm.train()
