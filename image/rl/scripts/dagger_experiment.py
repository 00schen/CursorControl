import rlkit.torch.pytorch_util as ptu
from rlkit.envs.make_env import make
from rlkit.torch.networks import Mlp,ConcatMlp
from rlkit.torch.networks import Clamp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import rlkit.pythonplusplus as ppp

from rl.policies import BoltzmannPolicy,OverridePolicy,ComparisonMergePolicy,ArgmaxPolicy
from rl.path_collectors import FullPathCollector,CustomPathCollector
from rl.env_wrapper import default_overhead
from rl.simple_path_loader import SimplePathLoader
from rl.trainers import DisBCTrainer
from rl.balanced_interventions_replay_buffer import BalancedInterReplayBuffer

import os
import gtimer as gt
from pathlib import Path
from rlkit.launchers.launcher_util import setup_logger,reset_execution_environment
import rlkit.util.hyperparameter as hyp
import argparse
from copy import deepcopy
import numpy as np
import torch.optim as optim
import torch as th
from torch.nn import functional as F
import json

def experiment(variant):
	from  rlkit.core import logger

	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid'])

	if 'pretrain_file_path' not in variant.keys():
		obs_dim = env.observation_space.low.size
		action_dim = env.action_space.low.size
		M = variant["layer_size"]
		upper_q = variant['env_kwargs']['config']['reward_max']*variant['env_kwargs']['config']['step_limit']
		lower_q = variant['env_kwargs']['config']['reward_min']*variant['env_kwargs']['config']['step_limit']
		qf1 = Mlp(
			input_size=obs_dim,
			output_size=action_dim,
			hidden_sizes=[M,M,M,M],
			hidden_activation=F.leaky_relu,
			layer_norm=True,
			# output_activation=Clamp(max=0,min=-5e3),
			output_activation=Clamp(max=upper_q,min=lower_q),
		)
	else:
		pretrain_file_path = variant['pretrain_file_path']
		qf1 = th.load(pretrain_file_path,map_location=ptu.device)['qf1']
	eval_policy = ArgmaxPolicy(
		qf1,
	)
	# eval_path_collector = CustomPathCollector(
	eval_path_collector = FullPathCollector(
		env,
		eval_policy,
		save_env_in_snapshot=False
	)
	if not variant['exploration_argmax']:
		expl_policy = BoltzmannPolicy(
			qf1,
			logit_scale=variant['expl_kwargs']['logit_scale'])
	else:
		expl_policy = ArgmaxPolicy(
			qf1,
		)
	if variant['exploration_strategy'] == 'merge_arg':
		expl_policy = ComparisonMergePolicy(env.rng,expl_policy,env.oracle.size)
	elif variant['exploration_strategy'] == 'override':
		expl_policy = OverridePolicy(expl_policy,env.oracle.size)
	expl_path_collector = FullPathCollector(
		env,
		expl_policy,
		save_env_in_snapshot=False
	)
	trainer = DisBCTrainer(
		policy=qf1,
		**variant['trainer_kwargs']
	)	
	replay_buffer = EnvReplayBuffer(
		variant['replay_buffer_size'],
		env,
	)
	algorithm = TorchBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=env,
		evaluation_env=env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		**variant['algorithm_args']
	)
	algorithm.to(ptu.device)
	if variant.get('load_demos', False):
		path_loader = SimplePathLoader(
			demo_path=variant['demo_paths'],
			demo_path_proportion=variant['demo_path_proportions'],
			replay_buffer=replay_buffer,
		)
		path_loader.load_demos()
	from rlkit.core import logger
	if variant.get('render',False):
		env.render('human')
	algorithm.train()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name',)
	parser.add_argument('--exp_name', default='a-test')
	parser.add_argument('--no_render', action='store_false')
	parser.add_argument('--use_ray', action='store_true')
	parser.add_argument('--gpus', default=0, type=int)
	parser.add_argument('--per_gpu', default=1, type=int)
	args, _ = parser.parse_known_args()
	main_dir = str(Path(__file__).resolve().parents[2])
	print(main_dir)

	path_length = 400
	num_epochs = int(1000)
	variant = dict(
		from_pretrain=False,
		layer_size=128,
		exploration_argmax=True,
		exploration_strategy='override',
		expl_kwargs=dict(
			logit_scale=1000,
		),
		replay_buffer_size=int(2e4)*path_length,
		# intervention_prop=.5
		trainer_kwargs=dict(
		),
		algorithm_args=dict(
			batch_size=256,
			max_path_length=path_length,
			eval_path_length=400,
			num_epochs=num_epochs,
			num_eval_steps_per_epoch=400,
			num_expl_steps_per_train_loop=path_length,
			num_trains_per_train_loop=100,				
		),

		load_demos=True,
		# demo_paths=[os.path.join(main_dir,"demos",demo)\
		# 			for demo in os.listdir(os.path.join(main_dir,"demos")) if f"{args.env_name}_keyboard" in demo],
		demo_paths=[
					os.path.join(main_dir,"demos",f"{args.env_name}_model_noisy_9500_success.npy"),
					# os.path.join(main_dir,"demos",f"{args.env_name}_model_off_policy_5000_success_1.npy"),
					# os.path.join(main_dir,"demos",f"{args.env_name}_model_off_policy_4000_fail_1.npy"),
					],
		# demo_path_proportions=[1]*9,
		pretrain_rf=False,
		pretrain=True,

		env_kwargs={'config':dict(
			env_name=args.env_name,
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03,frame_skip=5),
			# env_kwargs=dict(path_length=path_length,frame_skip=5),

			oracle='model',
			oracle_kwargs=dict(),
			input_in_obs=True,
			action_type='disc_traj',
			smooth_alpha = .8,

			adapts = ['high_dim_user','reward'],
			apply_projection=False,
			space=0,
			num_obs=10,
			num_nonnoop=0,
			reward_max=0,
			reward_min=-1,
			input_penalty=1,
			reward_type='sparse',
		)},
	)
	search_space = {
		'seedid': [2000,2002],

		'env_kwargs.config.oracle_kwargs.threshold': [.5],
		'env_kwargs.config.state_type': [2],

		'demo_path_proportions':[[100,],[10,],[1,]],
		'trainer_kwargs.policy_lr': [1e-3],
		'trainer_kwargs.use_mixup': [True],
	}


	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)
	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	def process_args(variant):
		variant['qf_lr'] = variant['trainer_kwargs']['policy_lr']
		variant['env_kwargs']['config']['seedid'] = variant['seedid']
		if not args.use_ray:
			variant['render'] = args.no_render

	if args.use_ray:
		import ray
		from ray.util import ActorPool
		from itertools import cycle,count
		ray.init(temp_dir='/tmp/ray_exp', num_gpus=args.gpus)

		@ray.remote
		class Iterators:
			def __init__(self):
				self.run_id_counter = count(0)
			def next(self):
				return next(self.run_id_counter)
		iterator = Iterators.options(name="global_iterator").remote()

		@ray.remote(num_cpus=1,num_gpus=1/args.per_gpu if args.gpus else 0)
		class Runner:
			def new_run(self,variant):
				gt.reset_root()
				ptu.set_gpu_mode(True)
				process_args(variant)
				iterator = ray.get_actor("global_iterator")
				run_id = ray.get(iterator.next.remote())
				save_path = os.path.join(main_dir,'logs')
				reset_execution_environment()
				setup_logger(exp_prefix=args.exp_name,variant=variant,base_log_dir=save_path,exp_id=run_id,)
				experiment(variant)
			def resume_run(self,variant):
				gt.reset_root()
				ptu.set_gpu_mode(True)
				process_args(variant)
				iterator = ray.get_actor("global_iterator")
				run_id = ray.get(iterator.next.remote())
				pretrain_path = os.path.join(main_dir,'logs',variant['pretrained_exp'])
				pretrain_path = str(sorted(Path(pretrain_path).iterdir())[run_id])
				with open(os.path.join(pretrain_path,'variant.json')) as variant_file:
					variant = json.load(variant_file)
				variant['pretrain_file_path'] = os.path.join(pretrain_path,'pretrain.pkl')
				save_path = os.path.join(main_dir,'logs')
				reset_execution_environment()
				setup_logger(exp_prefix=args.exp_name,variant=variant,base_log_dir=save_path,exp_id=run_id,snapshot_mode='gap_and_last')
				experiment(variant)
		runners = [Runner.remote() for i in range(args.gpus*args.per_gpu)]
		runner_pool = ActorPool(runners)
		if variant['from_pretrain']:
			list(runner_pool.map(lambda a,v: a.resume_run.remote(v), variants))
		else:
			list(runner_pool.map(lambda a,v: a.new_run.remote(v), variants))
	else:
		ptu.set_gpu_mode(False)
		process_args(variant)
		save_path = os.path.join(main_dir,'logs')
		reset_execution_environment()
		setup_logger(exp_prefix=args.exp_name,variant=variant,base_log_dir=save_path,exp_id=0,)
		experiment(variant)
