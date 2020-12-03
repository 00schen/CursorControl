import rlkit.torch.pytorch_util as ptu
from rlkit.envs.make_env import make
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.networks import Clamp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import BoltzmannPolicy,OverridePolicy,ComparisonMergePolicy
from rl.path_collectors import FullPathCollector
from rl.env_wrapper import default_overhead
from rl.simple_path_loader import SimplePathLoader

import os
from pathlib import Path
from rlkit.launchers.launcher_util import setup_logger,reset_execution_environment
import rlkit.util.hyperparameter as hyp
import argparse
import numpy as np
import torch.optim as optim

def experiment(variant):
	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid'])

	obs_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size
	M = variant["layer_size"]
	qf1 = ConcatMlp(
		input_size=obs_dim + action_dim,
		output_size=1,
		hidden_sizes=[M,M,M],
		output_activation=Clamp(max=0),
	)
	qf2 = ConcatMlp(
		input_size=obs_dim + action_dim,
		output_size=1,
		hidden_sizes=[M,M,M],
		output_activation=Clamp(max=0),
	)
	target_qf1 = ConcatMlp(
		input_size=obs_dim + action_dim,
		output_size=1,
		hidden_sizes=[M,M,M],
		output_activation=Clamp(max=0),
	)
	target_qf2 = ConcatMlp(
		input_size=obs_dim + action_dim,
		output_size=1,
		hidden_sizes=[M,M,M],
		output_activation=Clamp(max=0),
	)
	policy = TanhGaussianPolicy(
		obs_dim=obs_dim,
		action_dim=action_dim,
		hidden_sizes=[M,M,M],
	)
	eval_policy = MakeDeterministic(policy)
	eval_path_collector = FullPathCollector(
		env,
		eval_policy,
		save_env_in_snapshot=False,
	)
	expl_policy = policy
	if variant['exploration_strategy'] == 'merge_arg':
		expl_policy = ComparisonMergePolicy(env.rng,expl_policy,env.oracle.size)
	elif variant['exploration_strategy'] == 'override':
		expl_policy = OverridePolicy(env,expl_policy,env.oracle.size)
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
			replay_buffer=replay_buffer,
		)
		path_loader.load_demos()
	trainer = SACTrainer(
		env=env,
		policy=policy,
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
		for _ in range(variant['num_pretrain_loops']):
			train_data = replay_buffer.random_batch(variant['algorithm_args']['batch_size'])
			trainer.train(train_data)

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

	path_length = 200
	num_epochs = int(5e3)
	variant = dict(
		layer_size=512,
		exploration_strategy='',
		replay_buffer_size=(num_epochs//2)*path_length,
		trainer_kwargs=dict(
			discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
		),
		algorithm_args=dict(
			batch_size=1024,
			max_path_length=path_length,
			num_epochs=num_epochs,
			num_eval_steps_per_epoch=path_length,
			num_expl_steps_per_train_loop=path_length,
			num_trains_per_train_loop=50,				
		),
		load_demos=True,
		# demo_paths=[dict(
		# 			path=os.path.join(os.path.abspath(''),"demos",demo),
		# 			obs_dict=False,
		# 			is_demo=False,
		# 			train_split=1,
		# 			) for demo in os.listdir(os.path.join(os.path.abspath(''),"demos")) if f"{args.env_name}_keyboard" in demo],
		demo_paths=[os.path.join(main_dir,"demos",demo)\
					for demo in os.listdir(os.path.join(main_dir,"demos")) if f"{args.env_name}_model" in demo],
		pretrain=True,
		num_pretrain_loops=int(1e5),

		env_kwargs={'config':dict(
			env_name=args.env_name,
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03,frame_skip=5),
			# env_kwargs=dict(path_length=path_length,frame_skip=5),

			oracle='model',
			oracle_kwargs=dict(),
			action_type='trajectory',

			adapts = ['stack','reward'],
			space=0,
			num_obs=10,
			num_nonnoop=10,
			reward_max=0,
			reward_min=-np.inf,
			input_penalty=1,
		)},
	)

	search_space = {
		'seedid': [2000,2001,2002],

		'env_kwargs.config.smooth_alpha': [.8,],
		'env_kwargs.config.oracle_kwargs.threshold': [.8,.5,.2,0]
	}

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)
	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	def process_args(variant):
		variant['env_kwargs']['config']['seedid'] = variant['seedid']
		if not args.use_ray:
			variant['render'] = args.no_render
			variant['algorithm_args']['num_eval_steps_per_epoch'] = 0

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
		run_id=str(current_time)
		save_path = os.path.join(main_dir,'logs')
		setup_logger(exp_prefix=args.exp_name,variant=variant,base_log_dir=save_path,exp_id=run_id)
		process_args(variant)
		experiment(variant)
