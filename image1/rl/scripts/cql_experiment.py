import rlkit.torch.pytorch_util as ptu
from rlkit.envs.make_env import make
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.core import np_to_pytorch_batch
from rl.policies import OverridePolicy,ComparisonMergePolicy,MaxQPolicy
from rl.path_collectors import FullPathCollector,CustomPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.trainers import BCTrainer,CQLTrainer

import argparse, os
from pathlib import Path
from rlkit.launchers.launcher_util import setup_logger,reset_execution_environment
import rlkit.util.hyperparameter as hyp
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

def experiment(variant):
	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid'])
	
	obs_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size
	M = variant["layer_size"]
	qf1 = ConcatMlp(
		input_size=obs_dim + action_dim,
		output_size=1,
		hidden_sizes=[M,M,M,M],
		hidden_activation=F.leaky_relu,
		layer_norm=True,
	)
	qf2 = ConcatMlp(
		input_size=obs_dim + action_dim,
		output_size=1,
		hidden_sizes=[M,M,M,M],
		hidden_activation=F.leaky_relu,
		layer_norm=True,
	)
	target_qf1 = ConcatMlp(
		input_size=obs_dim + action_dim,
		output_size=1,
		hidden_sizes=[M,M,M,M],
		hidden_activation=F.leaky_relu,
		layer_norm=True,
	)
	target_qf2 = ConcatMlp(
		input_size=obs_dim + action_dim,
		output_size=1,
		hidden_sizes=[M,M,M,M],
		hidden_activation=F.leaky_relu,
		layer_norm=True,
	)
	policy = TanhGaussianPolicy(
		obs_dim=obs_dim,
		action_dim=action_dim,
		hidden_sizes=[M,M,M,M],
	)
	eval_policy = policy
	eval_path_collector = CustomPathCollector(
		env,
		eval_policy,
		save_env_in_snapshot=False,
	)
	# expl_policy = MaxQPolicy(policy,qf1)
	expl_policy = policy
	if variant['exploration_strategy'] == 'merge_arg':
		expl_policy = ComparisonMergePolicy(env.rng,expl_policy,env.oracle.size)
	elif variant['exploration_strategy'] == 'override':
		expl_policy = OverridePolicy(expl_policy,env.oracle.size)
	expl_path_collector = FullPathCollector(
		env,
		expl_policy,
		save_env_in_snapshot=False
	)
	cql_trainer = CQLTrainer(
			env=env,
			policy=policy,
			qf1=qf1,
			qf2=qf2,
			target_qf1=target_qf1,
			target_qf2=target_qf2,
			**variant['trainer_kwargs']
		)
	# trainer = SACTrainer(
	# 	env=env,
	# 	policy=policy,
	# 	qf1=qf1,
	# 	qf2=qf2,
	# 	target_qf1=target_qf1,
	# 	target_qf2=target_qf2,
	# 	**variant['sac_kwargs']
	# )
	replay_buffer = EnvReplayBuffer(
		variant['replay_buffer_size'],
		env,
	)
	algorithm = TorchBatchRLAlgorithm(
		trainer=cql_trainer,
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
	if variant['pretrain']:
		from tqdm import tqdm
		bc_trainer = BCTrainer(policy)
		for _ in tqdm(range(variant['num_pretrain_loops']),miniters=10,mininterval=10):
			bc_batch = replay_buffer.random_batch(variant['algorithm_args']['batch_size'])
			bc_trainer.pretrain(bc_batch)
		for _ in tqdm(range(variant['num_pretrain_loops']),miniters=10,mininterval=10):
			batch = replay_buffer.random_batch(variant['algorithm_args']['batch_size'])
			cql_trainer.train(batch)
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

	path_length = 400
	num_epochs = int(1e4)
	variant = dict(
		layer_size=512,
		exploration_strategy='',
		replay_buffer_size=10000*path_length,
		trainer_kwargs=dict(
			discount=0.999,
			reward_scale=1.0,

			policy_lr=1e-3,
			qf_lr=1e-4,
			optimizer_class=optim.Adam,

			soft_target_tau=1e-2,
			plotter=None,
			render_eval_paths=False,

			use_automatic_entropy_tuning=True,
			target_entropy=None,
			policy_eval_start=0,
			num_qs=2,

			# CQL
			min_q_version=3,
			# temp=1.0,
			# min_q_weight=1.0,

			## sort of backup
			# max_q_backup=False,
			deterministic_backup=True,
			num_random=10,
			with_lagrange=False,
			lagrange_thresh=0.0,
		),
		sac_kwargs=dict(
			discount=0.999,
			reward_scale=1.0,

			policy_lr=1e-3,
			qf_lr=1e-4,
			optimizer_class=optim.Adam,

			soft_target_tau=1e-2,
			target_update_period=1,
			plotter=None,
			render_eval_paths=False,

			use_automatic_entropy_tuning=True,
			target_entropy=None,
		),
		algorithm_args=dict(
			batch_size=256,
			max_path_length=path_length,
			eval_path_length=1,
			num_epochs=num_epochs,
			num_eval_steps_per_epoch=1,
			num_expl_steps_per_train_loop=path_length,
			# num_trains_per_train_loop=100,				
		),

		load_demos=True,
		demo_paths=[os.path.join(main_dir,"demos",f"{args.env_name}_model_off_policy_5000.npy"),
					os.path.join(main_dir,"demos",f"{args.env_name}_model_on_policy_5000.npy")],
		demo_path_proportions=[1,1],
		pretrain=True,
		num_pretrain_loops=int(1e4),

		env_kwargs={'config':dict(
			env_name=args.env_name,
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03,frame_skip=5),
			# env_kwargs=dict(path_length=path_length,frame_skip=5),

			oracle='model',
			oracle_kwargs=dict(),
			action_type='trajectory',
			smooth_alpha = .8,

			adapts = ['high_dim_user','reward'],
			space=0,
			num_obs=10,
			num_nonnoop=10,
			reward_max=0,
			reward_min=-1,
			input_penalty=1,
			reward_type='user_penalty',
		)},
	)
	search_space = {
		'seedid': [2000],

		'trainer_kwargs.temp': [.5,.1],
		'trainer_kwargs.min_q_weight': [2,.5],
		'trainer_kwargs.max_q_backup': [False,True],
		'env_kwargs.config.apply_projection': [False],
		'algorithm_args.num_trains_per_train_loop': [5],
		'env_kwargs.config.oracle_kwargs.threshold': [.8,],
	}

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)
	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	def process_args(variant):
		variant['qf_lr'] = variant['trainer_kwargs']['qf_lr']
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
							)
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
		