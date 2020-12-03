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
