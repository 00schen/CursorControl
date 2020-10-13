import torch as th
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from railrl.samplers.data_collector import MdpPathCollector
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from railrl.torch.networks import Mlp
from railrl.torch.dqn.double_dqn import DoubleDQNTrainer

from railrl.demos.source.mdp_path_loader import MDPPathLoader

from railrl.core import logger
import railrl.torch.pytorch_util as ptu
from railrl.envs.make_env import make
from railrl.misc.eval_util import create_stats_ordered_dict

from collections import OrderedDict
from collections import deque
from railrl.core.timer import timer

from replay_buffers import PavlovReplayBuffer
from utils import rng
from agents import DemonstrationPolicy,ArgmaxDiscretePolicy,BoltzmannPolicy,HybridPolicy,TranslationPolicy

class FullTrajPathCollector(MdpPathCollector):
	def collect_new_paths(
			self,
			max_path_length,
			num_steps,
	):
		paths = []
		num_steps_collected = 0
		while num_steps_collected < num_steps:
			path = self._rollout_fn(
				self._env,
				self._policy,
				max_path_length=max_path_length,
				render=self._render,
				render_kwargs=self._render_kwargs,
			)
			path_len = len(path['actions'])
			num_steps_collected += path_len
			paths.append(path)
		self._num_paths_total += len(paths)
		self._num_steps_total += num_steps_collected
		self._epoch_paths.extend(paths)
		return paths

class PavlovBatchRLAlgorithm(TorchBatchRLAlgorithm):
	def __init__(self, *args, **kwargs):
		self.dump_tabular = kwargs.pop('dump_tabular',True)
		super().__init__(*args,**kwargs)

	def train(self):
		timer.return_global_times = True
		for _ in range(self.num_epochs):
			self._begin_epoch()
			timer.start_timer('saving')
			logger.save_itr_params(self.epoch, self._get_snapshot())
			timer.stop_timer('saving')
			log_dict, _ = self._train()
			logger.record_dict(log_dict)
			if self.dump_tabular:
				logger.dump_tabular(with_prefix=True, with_timestamp=False)
			self._end_epoch()
		logger.save_itr_params(self.epoch, self._get_snapshot())


	def _train(self):
		done = (self.epoch == self.num_epochs)
		if done:
			return OrderedDict(), done

		if self.epoch == 0 and self.min_num_steps_before_training > 0:
			init_expl_paths = self.expl_data_collector.collect_new_paths(
				self.max_path_length,
				self.min_num_steps_before_training,
			)
			print(init_expl_paths['actions'].mean(axis=0))
			self.replay_buffer.add_paths(init_expl_paths)
			self.expl_data_collector.end_epoch(-1)

		timer.start_timer('evaluation sampling')
		if self.epoch % self._eval_epoch_freq == 0:
			self.eval_data_collector.collect_new_paths(
				self.max_path_length,
				self.num_eval_steps_per_epoch,
			)
		timer.stop_timer('evaluation sampling')

		if not self._eval_only:
			for _ in range(self.num_train_loops_per_epoch):
				timer.start_timer('exploration sampling', unique=False)
				new_expl_paths = self.expl_data_collector.collect_new_paths(
					self.max_path_length,
					self.num_expl_steps_per_train_loop,
				)
				timer.stop_timer('exploration sampling')

				timer.start_timer('replay buffer data storing', unique=False)
				self.replay_buffer.add_paths(new_expl_paths)
				timer.stop_timer('replay buffer data storing')

				timer.start_timer('training', unique=False)
				for _ in range(self.num_trains_per_train_loop):
					train_data = self.replay_buffer.random_batch(self.batch_size)
					self.trainer.train(train_data)
				timer.stop_timer('training')
		log_stats = self._get_diagnostics()
		return log_stats, False

class DQNPavlovTrainer(DoubleDQNTrainer):
	def __init__(self,qf1,target_qf1,qf2,target_qf2,**kwargs):
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

	def train_from_torch(self, batch):
		rewards = batch['rewards']
		terminals = batch['terminals']
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
		y_target = rewards + (1. - terminals) * self.discount * target_q_values
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
		]
		return nets

	def get_snapshot(self):
		return dict(
			qf1 = self.qf1,
			target_qf1 = self.target_qf1,
			qf2 = self.qf2,
			target_qf2 = self.target_qf2,
			# pf = self.pf,
		)

from types import MethodType
def demonstration_factory(base):
	class DemonstrationEnv(base):
		def __init__(self,config):
			super().__init__(config)
			self.target_index = 0
			self.num_targets = self.env.num_targets

		def next_target(self):
			self.target_index += 1
		def reset_target(self):
			self.target_index = 0

		def reset(self):
			target_index = self.target_index
			def generate_target(self,index):
				nonlocal target_index
				self.__class__.generate_target(self,target_index)
			self.env.generate_target = MethodType(generate_target,self.env)
			return super().reset()
	return DemonstrationEnv
def collect_demonstrations(variant):
	env_class = variant['env_class']
	env_kwargs = variant.get('env_kwargs', {})
	env_kwargs['config']['factories'] += demonstration_factory
	env = make(None, env_class, env_kwargs, False)
	env.seed(variant['seedid'])

	path_collector = FullTrajPathCollector(
		env,
		DemonstrationPolicy(env,lower_p=.8,upper_p=1,traj_len=env_kwargs['config']['traj_len']),
	)

	if variant.get('render',False):
		env.render('human')
	demo_kwargs = variant.get('demo_kwargs')
	paths = []
	num_successes = 0
	fail_paths = deque([],int(demo_kwargs['min_successes']*(1/demo_kwargs['min_success_rate']-1)))
	while num_successes < demo_kwargs['min_successes']:
		while env.target_index < env.num_targets:
			collected_paths = path_collector.collect_new_paths(
				demo_kwargs['path_length'],
				demo_kwargs['path_length'],
			)
			success = False
			for path in collected_paths:
				if path['env_infos'][-1]['task_success']:
					paths.append(path)
					success = True
					num_successes += 1
				else:
					fail_paths.append(path)
			if success:
				env.next_target()
			print("total paths collected: ", len(paths), "total successes: ", num_successes)
		env.reset_target()
	paths.extend(fail_paths)

	return paths

def eval_exp(variant):
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})
	eval_env = make(None, env_class, env_kwargs, False)
	# eval_env.seed(variant['seedid'])
	eval_env.seed(1999)

	file_name = os.path.join(variant['save_path'],'params.pkl')
	qf1 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf1']
	qf2 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf2']

	policy_kwargs = variant['policy_kwargs']
	policy = ArgmaxDiscretePolicy(
		qf1=qf1,
		qf2=qf2,
		logit_scale=1000,
		**policy_kwargs,
	)

	# eval_policy = HybridAgent(policy)
	eval_policy = policy
	eval_path_collector = FullTrajPathCollector(
		eval_env,
		eval_policy,
	)

	if variant.get('render',False):
		eval_env.render('human')
	eval_collected_paths = eval_path_collector.collect_new_paths(
		variant['algorithm_args']['max_path_length'],
		variant['algorithm_args']['num_eval_steps_per_epoch']*10,
	)

	np.save(os.path.join(variant['save_path'],'evaluated_paths'), eval_collected_paths)

def resume_exp(variant):
	normalize_env = variant.get('normalize_env', True)
	env_id = variant.get('env_id', None)
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})

	expl_env = make(env_id, env_class, env_kwargs, normalize_env)
	expl_env.seed(variant['seedid'])

	file_name = os.path.join(variant['file_name'],'params.pkl')
	qf1 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf1']
	target_qf1 = th.load(file_name,map_location=th.device("cpu"))['trainer/target_qf1']
	qf2 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf2']
	target_qf2 = th.load(file_name,map_location=th.device("cpu"))['trainer/target_qf2']

	policy_kwargs = variant['policy_kwargs']
	policy = ArgmaxDiscretePolicy(
		qf1=qf1,
		qf2=qf2,
		**policy_kwargs,
	)

	eval_policy = policy
	eval_path_collector = FullTrajPathCollector(
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
			expl_policy = BoltzmannPolicy(qf1,qf2,logit_scale=exploration_kwargs['logit_scale'],**policy_kwargs)
		else:
			error


	trainer_class = variant.get("trainer_class", DQNPavlovTrainer)
	if trainer_class == DQNPavlovTrainer:
		trainer = trainer_class(
			qf1=qf1,
			target_qf1=target_qf1,
			qf2=qf2,
			target_qf2=target_qf2,
			**variant['trainer_kwargs']
		)
	else:
		error

	expl_path_collector = FullTrajPathCollector(
		expl_env,
		expl_policy,
	)
	algorithm = PavlovBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=expl_env,
		evaluation_env=expl_env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		**variant['algorithm_args']
	)
	algorithm.to(ptu.device)
	algorithm.train()

def experiment(variant):
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})
	expl_env = make(None, env_class, env_kwargs, False)
	expl_env.seed(variant['seedid'])

	if variant.get('add_env_demos', False):
		variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])
	if variant.get('add_env_offpolicy_data', False):
		variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

	path_loader_kwargs = variant.get("path_loader_kwargs", {})
	action_dim = expl_env.action_space.low.size

	obs_dim = expl_env.observation_space.low.size
	qf_kwargs = variant.get("qf_kwargs", {})
	qf1 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_activation=F.leaky_relu,
		**qf_kwargs
	)
	target_qf1 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_activation=F.leaky_relu,
		**qf_kwargs
	)
	qf2 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_activation=F.leaky_relu,
		**qf_kwargs
	)
	target_qf2 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_activation=F.leaky_relu,
		**qf_kwargs
	)

	policy_kwargs = variant['policy_kwargs']
	policy = ArgmaxDiscretePolicy(
		qf1=qf1,
		qf2=qf2,
		**policy_kwargs,
	)

	eval_policy = TranslationPolicy(HybridPolicy(policy))
	eval_path_collector = FullTrajPathCollector(
		expl_env,
		eval_policy,
	)

	exploration_kwargs =  variant.get('exploration_kwargs', {})
	if exploration_kwargs:
		exploration_strategy = exploration_kwargs.get("strategy", None)
		if exploration_strategy is None:
			pass
		elif exploration_strategy == 'boltzmann':
			expl_policy = BoltzmannPolicy(qf1,qf2,
										logit_scale=exploration_kwargs['logit_scale'],
										**policy_kwargs)
		else:
			error
	expl_policy = TranslationPolicy(HybridPolicy(expl_policy))

	replay_buffer_kwargs = variant.get('replay_buffer_kwargs',{})
	replay_buffer_kwargs.update({'env':expl_env})
	replay_buffer = variant.get('replay_buffer_class', PavlovReplayBuffer)(
		**replay_buffer_kwargs,
	)

	trainer_class = variant.get("trainer_class", DQNPavlovTrainer)
	if trainer_class == DQNPavlovTrainer:
		trainer = trainer_class(
			qf1=qf1,
			target_qf1=target_qf1,
			qf2=qf2,
			target_qf2=target_qf2,
			**variant['trainer_kwargs']
		)
	else:
		error

	expl_path_collector = FullTrajPathCollector(
		expl_env,
		expl_policy,
	)
	algorithm = PavlovBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=expl_env,
		evaluation_env=expl_env,
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
		if variant.get('render',False):
			expl_env.render('human')
		algorithm.train()
