import rlkit.torch.pytorch_util as ptu
from rlkit.envs.make_env import make
from rlkit.torch.networks import Mlp
from rlkit.torch.networks import Clamp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import rlkit.pythonplusplus as ppp

from rl.policies import BoltzmannPolicy,OverridePolicy,ComparisonMergePolicy,ArgmaxPolicy
from rl.path_collectors import FullPathCollector,CustomPathCollector
from rl.env_wrapper import default_overhead
from rl.simple_path_loader import SimplePathLoader
from rl.trainers import DDQNCQLTrainer

import os
from pathlib import Path
from rlkit.launchers.launcher_util import setup_logger,reset_execution_environment
import rlkit.util.hyperparameter as hyp
import argparse
from copy import deepcopy
import numpy as np
import torch.optim as optim
import torch as th

def experiment(variant):
	from  rlkit.core import logger

	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid'])

	if not variant['from_pretrain']:
		obs_dim = env.observation_space.low.size
		action_dim = env.action_space.low.size
		M = variant["layer_size"]
		qf1 = Mlp(
			input_size=obs_dim,
			output_size=action_dim,
			hidden_sizes=[M,M,M],
			output_activation=Clamp(max=0),
		)
		qf2 = Mlp(
			input_size=obs_dim,
			output_size=action_dim,
			hidden_sizes=[M,M,M],
			output_activation=Clamp(max=0),
		)
		target_qf1 = Mlp(
			input_size=obs_dim,
			output_size=action_dim,
			hidden_sizes=[M,M,M],
			output_activation=Clamp(max=0),
		)
		target_qf2 = Mlp(
			input_size=obs_dim,
			output_size=action_dim,
			hidden_sizes=[M,M,M],
			output_activation=Clamp(max=0),
		)
	else:
		pretrain_file_path = variant['pretrain_file_path']
		qf1 = th.load(pretrain_file_path,map_location=th.device("cpu"))['qf1']
		qf2 = th.load(pretrain_file_path,map_location=th.device("cpu"))['qf1']
		target_qf1 = th.load(pretrain_file_path,map_location=th.device("cpu"))['target_qf1']
		target_qf2 = th.load(pretrain_file_path,map_location=th.device("cpu"))['target_qf2']
	eval_policy = ArgmaxPolicy(
		qf1,qf2,
	)
	# eval_path_collector = CustomPathCollector(
	eval_path_collector = FullPathCollector(
		env,
		eval_policy,
		save_env_in_snapshot=False
	)
	if not variant['exploration_argmax']:
		expl_policy = BoltzmannPolicy(
			qf1,qf2,
			logit_scale=variant['expl_kwargs']['logit_scale'])
	else:
		expl_policy = ArgmaxPolicy(
			qf1,qf2
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
	replay_buffer = EnvReplayBuffer(
		variant['replay_buffer_size'],
		env,
	)
	if variant.get('load_demos', False):
		path_loader = SimplePathLoader(
			demo_path=variant['demo_paths'],
			demo_path_proportion=variant['demo_path_proportions'],
			replay_buffer=replay_buffer,
		)
		path_loader.load_demos()
	trainer = DDQNCQLTrainer(
		qf1=qf1,
		qf2=qf2,
		target_qf1=target_qf1,
		target_qf2=target_qf2,
		**variant['trainer_kwargs']
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
	if variant['pretrain']:
		from tqdm import tqdm
		for _ in tqdm(range(variant['num_pretrain_loops']),miniters=10,mininterval=10):
			train_data = replay_buffer.random_batch(variant['algorithm_args']['batch_size'])
			trainer.train(train_data)
		pretrain_file_path = os.path.join(logger.get_snapshot_dir(), 'pretrain.pkl')
		th.save(trainer.get_snapshot(), pretrain_file_path)
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
	num_epochs = 10
	variant = dict(
		from_pretrain=False,
		# pretrain_file_path=os.path.join(main_dir,'logs','pretrain-cql','pretrain_cql_2020_12_07_17_15_47_0000--s-0','pretrain.pkl'),
		layer_size=512,
		exploration_argmax=True,
		exploration_strategy='',
		expl_kwargs=dict(
			logit_scale=1000,
		),
		replay_buffer_size=10000*path_length,
		trainer_kwargs=dict(
			qf_lr=1e-4,
			soft_target_tau=1e-2,
			target_update_period=1,
			qf_criterion=None,

			discount=0.999,
			reward_scale=1.0,

			# temp=1.0,
			# min_q_weight=1.0,
		),
		algorithm_args=dict(
			batch_size=256,
			max_path_length=path_length,
			eval_path_length=1,
			num_epochs=num_epochs,
			num_eval_steps_per_epoch=1,
			num_expl_steps_per_train_loop=path_length,
			num_trains_per_train_loop=50,				
		),

		load_demos=True,
		# demo_paths=[os.path.join(main_dir,"demos",demo)\
		# 			for demo in os.listdir(os.path.join(main_dir,"demos")) if f"{args.env_name}_keyboard" in demo],
		demo_paths=[os.path.join(main_dir,"demos",f"{args.env_name}_model_off_policy_5000.npy"),
					os.path.join(main_dir,"demos",f"{args.env_name}_model_on_policy_5000.npy")],
		# demo_path_proportions=[.5,.5],
		pretrain=True,
		num_pretrain_loops=int(1e4),

		env_kwargs={'config':dict(
			env_name=args.env_name,
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03,frame_skip=5),
			# env_kwargs=dict(path_length=path_length,frame_skip=5),

			oracle='model',
			oracle_kwargs=dict(),
			action_type='disc_traj',
			smooth_alpha = .8,

			adapts = ['high_dim_user','reward'],
			space=0,
			num_obs=10,
			num_nonnoop=10,
			reward_max=0,
			reward_min=-1,
			input_penalty=1,
			# sparse_reward=False,
		)},
	)
	search_space = {
		'seedid': [2000,2001],

		'trainer_kwargs.temp': [1],
		'trainer_kwargs.min_q_weight': [1],
		'env_kwargs.config.oracle_kwargs.threshold': [.5,],
		'env_kwargs.config.apply_projection': [False],
		# 'env_kwargs.config.oracle_kwargs.epsilon': [0,.25],
		# 'env_kwargs.config.oracle_kwargs.threshold': [.2,0]

		'env_kwargs.config.sparse_reward': [False,True],
		'demo_path_proportions': [[0,1],[.2,.8],[.4,.6],[.6,.4],[.8,.2],[1,0]]
		# 'demo_paths':[[os.path.join(main_dir,"demos",f"{args.env_name}_model_long.npy")],[os.path.join(main_dir,"demos",f"{args.env_name}_model_off_policy_500.npy")]],
	}
	# True: 1,1 False: .1,1
	# other_spaces = [
	# 	{
	# 		'env_kwargs.config.sparse_reward': False,
	# 		'trainer_kwargs.temp': 1e-1,
	# 	},
	# 	{
	# 		'env_kwargs.config.sparse_reward': True,
	# 		'trainer_kwargs.temp': 1,
	# 	},
	# ]
	# other_spaces = [ppp.dot_map_dict_to_nested_dict(space) for space in other_spaces]


	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)
	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)
	# variants = sum([[ppp.merge_recursive_dicts(
	# 					deepcopy(variant),
	# 					deepcopy(space),
	# 					ignore_duplicate_keys_in_second_dict=True,
	# 				) for variant in variants] for space in other_spaces],[])

	def process_args(variant):
		variant['trainer_kwargs']['learning_rate'] = variant['trainer_kwargs'].pop('qf_lr')
		variant['qf_lr'] = variant['trainer_kwargs']['learning_rate']
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
			def run(self,variant):
				import gtimer as gt
				gt.reset_root()
				ptu.set_gpu_mode(True)
				process_args(variant)
				iterator = ray.get_actor("global_iterator")
				run_id = ray.get(iterator.next.remote())
				save_path = os.path.join(main_dir,'logs')
				reset_execution_environment()
				setup_logger(exp_prefix=args.exp_name,variant=variant,base_log_dir=save_path,exp_id=run_id,
							tensorboard=True,)
				experiment(variant)
		runners = [Runner.remote() for i in range(args.gpus*args.per_gpu)]
		runner_pool = ActorPool(runners)
		list(runner_pool.map(lambda a,v: a.run.remote(v), variants))
	else:
		import time
		current_time = time.time_ns()
		variant = variants[0]
		run_id=0
		save_path = os.path.join(main_dir,'logs')
		setup_logger(exp_prefix=args.exp_name,variant=variant,base_log_dir=save_path,exp_id=run_id)
		process_args(variant)
		experiment(variant)
