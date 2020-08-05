import gym
import torch
import numpy as np

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.samplers.data_collector import MdpPathCollector
from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.torch.torch_rl_algorithm import (
	TorchBatchRLAlgorithm,
	TorchOnlineRLAlgorithm,
)

from railrl.torch.networks import Mlp
# from railrl.torch.dqn.policy import ArgmaxDiscretePolicy
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
class ArgmaxDiscretePolicy(PyTorchModule):
	def __init__(self, qf):
		super().__init__()
		self.qf = qf

	def get_action(self, obs):
		obs = torch.from_numpy(obs).float().cuda()
		q_values = self.qf(obs)
		action = F.one_hot(q_values.argmax(0,keepdim=True),list(q_values.size())[0]).cpu().flatten().detach().numpy()
		return action, {}

	def reset(self):
		pass

class NewDoubleDQNTrainer(DoubleDQNTrainer):
	def __init__(self,qf,target_qf,pretrain_steps=int(5e4),**kwargs):
		super().__init__(qf,target_qf,**kwargs)
		self.pretrain_steps = pretrain_steps

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

	@property
	def networks(self):
		nets = [
			self.qf,
			self.target_qf,
		]
		return nets

def experiment(variant):
	normalize_env = variant.get('normalize_env', True)
	env_id = variant.get('env_id', None)
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})

	expl_env = make(env_id, env_class, env_kwargs, normalize_env)
	eval_env = make(env_id, env_class, env_kwargs, normalize_env)

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
	qf = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		**qf_kwargs
	)
	target_qf = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		**qf_kwargs
	)

	policy_class = variant.get("policy_class", ArgmaxDiscretePolicy)
	policy_kwargs = variant['policy_kwargs']
	policy = policy_class(
		qf=qf,
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
		exploration_strategy = exploration_kwargs.get("strategy", 'gauss_eps')
		if exploration_strategy is None:
			pass
		elif exploration_strategy == 'gauss_eps':
			es = GaussianAndEpislonStrategy(
				action_space=expl_env.action_space,
				max_sigma=exploration_kwargs['noise'],
				min_sigma=exploration_kwargs['noise'],  # constant sigma
				epsilon=0,
			)
			expl_policy = PolicyWrappedWithExplorationStrategy(
				exploration_strategy=es,
				policy=expl_policy,
			)
		else:
			error

	main_replay_buffer_kwargs=dict(
		max_replay_buffer_size=variant['replay_buffer_size'],
		env=expl_env,
	)
	replay_buffer_kwargs = dict(
		max_replay_buffer_size=variant['replay_buffer_size'],
		env=expl_env,
	)

	replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
		**main_replay_buffer_kwargs,
	)
	
	trainer_class = variant.get("trainer_class", NewDoubleDQNTrainer)
	trainer = trainer_class(
		# env=eval_env,
		# policy=policy,
		qf=qf,
		target_qf=target_qf,
		**variant['trainer_kwargs']
	)
	if variant['collection_mode'] == 'online':
		expl_path_collector = MdpStepCollector(
			expl_env,
			policy,
		)
		algorithm = TorchOnlineRLAlgorithm(
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
	else:
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

	demo_train_buffer = EnvReplayBuffer(
		**replay_buffer_kwargs,
	)
	demo_test_buffer = EnvReplayBuffer(
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
	if variant.get('pretrain_rl', False):
		trainer.pretrain()
	if variant.get('train_rl', True):
		algorithm.train()
		