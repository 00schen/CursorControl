import torch as th
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import numpy as np
import os

from railrl.samplers.data_collector import MdpPathCollector
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from railrl.core import logger
from railrl.torch.core import np_to_pytorch_batch
import railrl.torch.pytorch_util as ptu
from railrl.envs.make_env import make
from railrl.misc.eval_util import create_stats_ordered_dict

from collections import OrderedDict
from collections import deque
from copy import deepcopy

from utils import *
from policies import DemonstrationPolicy,BoltzmannPolicy,OverridePolicy,ComparisonMergeArgPolicy
from full_path_collector import FullPathCollector
			
def collect_demonstrations(variant):
	env_class = variant['env_class']
	env_kwargs = variant.get('env_kwargs', {})
	env = make(None, env_class, env_kwargs, False)
	env.seed(variant['seedid'])

	path_collector = FullPathCollector(
		env,
		DemonstrationPolicy(env,p=.8),
	)

	if variant.get('render',False):
		env.render('human')
	demo_kwargs = variant.get('demo_kwargs')
	paths = []
	while len(paths) < demo_kwargs['num_episodes']:
		target_index = 0
		while target_index < env.base_env.num_targets:
			collected_paths = path_collector.collect_new_paths(
				demo_kwargs['path_length'],
				demo_kwargs['path_length'],
			)
			success_found = False
			for path in collected_paths:
				if path['env_infos'][-1]['task_success'] or (not demo_kwargs['only_success']):
					paths.append(path)
					success_found = True
			if success_found:
				target_index += 1	
				env.base_env.set_target_index(target_index)
			print("total paths collected: ", len(paths))
	return paths

def eval_exp(variant):
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})
	env = make(None, env_class, env_kwargs, False)
	env.seed(variant['seedid']+10)

	file_name = os.path.join(variant['eval_path'],'params.pkl')
	qf1 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf1']
	qf2 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf2']

	policy_kwargs = variant['policy_kwargs']
	policy = BoltzmannPolicy(
		qf1=qf1,
		qf2=qf2,
		**policy_kwargs,
	)
	eval_path_collector = FullPathCollector(
		env,
		policy,
	)

	if variant.get('render',False):
		env.render('human')
	eval_collected_paths = eval_path_collector.collect_new_paths(
		variant['algorithm_args']['max_path_length'],
		variant['algorithm_args']['num_eval_steps_per_epoch']*10,
	)

	np.save(os.path.join(variant['save_path'],'evaluated_paths'), eval_collected_paths)

def experiment(variant):
	env_class = variant.get('env_class', None)
	env_kwargs = variant.get('env_kwargs', {})
	env = make(None, env_class, env_kwargs, False)
	env.seed(variant['seedid'])

	qf_kwargs = variant.get("qf_kwargs", {})
	obs_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size
	qf_kwargs['hidden_activation'] = F.leaky_relu

	qf1b = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		**qf_kwargs
	)
	qf2b = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		**qf_kwargs
	)
	policy_kwargs = variant['policy_kwargs']
	eval_policy = ArgmaxDiscretePolicy(
		qf1b,qf2b,
		**policy_kwargs,)
	eval_policy = TranslationPolicy(env,eval_policy,**env_kwargs['config'])

	exploration_kwargs =  variant['exploration_kwargs']
	exploration_strategy = exploration_kwargs.get("strategy",'argmax')
	if exploration_strategy == 'boltzmann':
		expl_policy = BoltzmannPolicy(
			qf1a,qf2a,
			logit_scale=exploration_kwargs['logit_scale'],**policy_kwargs)
	elif exploration_strategy == 'merge_arg':
		expl_policy = ComparisonMergeArgPolicy(env,
			qf1a,qf2a,
			exploration_kwargs['alpha'],**policy_kwargs)
	else:
		error
	if exploration_kwargs['override']:
		expl_policy = OverridePolicy(env,expl_policy)
	expl_policy = TranslationPolicy(env,expl_policy,**env_kwargs['config'])

	qf_kwargs['obs_dim'] = env.observation_space.low.size
	qf_kwargs['action_dim'] = env.action_space.low.size
	if exploration_strategy == 'rnd':
		trainer = FullRNDTrainer(
			qf1a,qf2a,qf1b,qf2b,
			deepcopy(qf_kwargs),
			**variant['trainer_kwargs'])
	else:
		trainer = D2DQNTrainer(
			qf1b,qf2b,
			deepcopy(qf_kwargs),
			**variant['trainer_kwargs'])

	replay_buffer_kwargs = variant.get('replay_buffer_kwargs',{})
	replay_buffer_kwargs.update({'env':env})
	replay_buffer = variant.get('replay_buffer_class', AdaptReplayBuffer)(
		**replay_buffer_kwargs,
	)

	if variant.get('load_demos', False):
		path_loader_kwargs = variant.get("path_loader_kwargs", {})
		path_loader_class = variant.get('path_loader_class', AdaptPathLoader)
		path_loader = path_loader_class(trainer,
			replay_buffer=replay_buffer,
			demo_train_buffer=replay_buffer,
			demo_test_buffer=replay_buffer,
			**path_loader_kwargs
		)
		path_loader.load_demos()

	expl_path_collector = FullTrajPathCollector(
		env,
		expl_policy,
	)
	eval_path_collector = FullTrajPathCollector(
		env,
		eval_policy,
	)
	algorithm = PavlovBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=env,
		evaluation_env=env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		path_loader = path_loader if variant.get('load_demos', False) else None,
		**variant['algorithm_args']
	)
	algorithm.to(ptu.device)

	print(replay_buffer._top)
	if variant.get('render',False):
		env.render('human')
	algorithm.train()
