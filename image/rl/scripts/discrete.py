import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants
from railrl.torch.networks import Clamp

from discrete_experiment import experiment,eval_exp,collect_demonstrations
from envs import default_overhead,train_oracle_factory

import argparse
from copy import deepcopy,copy
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--env_name',)
parser.add_argument('--job',)
parser.add_argument('--exp_name', default='a-test')
parser.add_argument('--no_render', action='store_false')
parser.add_argument('--no_dump_tabular', action='store_false')
parser.add_argument('--use_ray', action='store_true')
parser.add_argument('--gpus', default=0, type=int)
parser.add_argument('--per_gpu', default=1, type=int)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
	path_length = 200
	num_epochs = int(5e3)
	variant = dict(
		algorithm_args=dict(
			num_epochs=num_epochs,
			num_eval_steps_per_epoch=path_length,
			num_trains_per_train_loop=50,
			num_expl_steps_per_train_loop=path_length*3//4,
			min_num_steps_before_training=0,

			batch_size=1024,
			max_path_length=path_length,
			pretrain=True,
			num_pretrains=int(1e5),
		),

		replay_buffer_kwargs=dict(
			# max_replay_buffer_size=int(1e4)*path_length,
			max_replay_buffer_size=500*path_length,
		),

		layer_size=512,

		trainer_kwargs=dict(
			discount=.995,
			learning_rate=5e-4,
			soft_target_tau=1e-3,
			target_update_period=10,
			reward_scale=1,
		),

		version="normal",
		collection_mode='batch',
		launcher_config=dict(
			exp_name=args.exp_name,
			mode='here_no_doodad',
			use_gpu=args.gpus,
		),
		logger_config=dict(
			snapshot_mode='last',
			# snapshot_gap=2,
			tensorboard=True,
		),

		path_loader_kwargs=dict(
			obs_key="state_observation",

			# demo_paths=[dict(
			# 			path=os.path.join(os.path.abspath(''),"demos",demo),
			# 			obs_dict=False,
			# 			is_demo=False,
			# 			train_split=1,
			# 			) for demo in os.listdir(os.path.join(os.path.abspath(''),"demos")) if f"{args.env_name}_keyboard" in demo],
			demo_paths=[dict(
						path=os.path.join(os.path.abspath(''),"demos",demo),
						obs_dict=False,
						is_demo=False,
						train_split=1,
						) for demo in os.listdir(os.path.join(os.path.abspath(''),"demos")) if f"{args.env_name}_model" in demo],
		),

		eval_path=os.path.join(os.path.abspath(''),"logs","testli-11","run1","id0"),


		load_demos=True,
		demo_kwargs=dict(
			only_success=True,
			num_episodes=500,
			path_length=path_length,
			# save_name=f"{args.env_name}_keyboardinput_"
			save_name=f"{args.env_name}_model_2000"
		)
	)
	config = dict(
		env_name=args.env_name,
		step_limit=path_length,
		env_kwargs=dict(success_dist=.03,frame_skip=5),
		# env_kwargs=dict(path_length=path_length,frame_skip=5),

		# factories=[train_oracle_factory],
		factories = [],
		oracle='keyboard',
		oracle_kwargs=dict(),
		action_type='disc_traj',
		traj_len=.1,

		# adapts = ['train'],
		adapts = ['burst','stack','reward'],
		space=0,
		num_obs=10,
		num_nonnoop=10,
		reward_max=0,
		reward_min=-np.inf,
		input_penalty=1,		
	)
	variant.update(dict(
		env_class=default_overhead,
		env_kwargs={'config':config},
	))

	search_space = {
		'seedid': [2000,2001,2002],
		# 'trainer_kwargs.learning_rate': [5e-4,],
		# 'trainer_kwargs.soft_target_tau': [1e-3],
		# 'env_kwargs.config.oracle_kwargs.blank': [1,],
		# 'env_kwargs.config.env_kwargs.success_dist': [.25,.1,],
		'env_kwargs.config.oracle_kwargs.threshold': [.2,0,-.2,],
		# 'env_kwargs.config.input_penalty': [.1,.25,.5,1],
		# 'env_kwargs.config.oracle_kwargs.threshold': [.3,.15],
		# 'env_kwargs.config.env_kwargs.success_dist': [.1,.15],
		'env_kwargs.config.smooth_alpha': [.8,],

		'exploration_kwargs.logit_scale': [1,100]

		# 'trainer_kwargs.cql_kwargs.re_shift': [0,10,50,100],
		# 'trainer_kwargs.cql_kwargs.re_scale': [.5,1,2,5],
		# 'trainer_kwargs.cql_kwargs.cql_alpha': [0,.2,.5,1]
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
			if args.job in ['exp']:
				variant['algorithm_args']['num_eval_steps_per_epoch'] = 0
				variant['algorithm_args']['dump_tabular'] = args.no_dump_tabular
			elif args.job in ['demo']:
				variant['demo_kwargs']['num_episodes'] = 10
			elif args.job in ['practice']:
				variant['demo_kwargs']['num_episodes'] = 10
		if args.job in ['demo']:
			variant['env_kwargs']['config']['adapts'] = []

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

		if args.job in ['exp',]:
			@ray.remote(num_cpus=1,num_gpus=1/args.per_gpu if args.gpus else 0)
			class Runner:
				def run(self,variant):
					iterator = ray.get_actor("global_iterator")
					run_id = ray.get(iterator.next.remote())
					variant['launcher_config']['gpu_id'] = 0
					variant['algorithm_args']['eval_path_name'] = run_id
					run_variants(experiment, [variant], process_args,run_id=run_id)
			runners = [Runner.remote() for i in range(args.gpus*args.per_gpu)]
			runner_pool = ActorPool(runners)
			list(runner_pool.map(lambda a,v: a.run.remote(v), variants))
		elif args.job in ['demo']:
			variant = variants[0]
			process_args(variant)
			@ray.remote(num_cpus=1,num_gpus=0)
			class Sampler:
				def sample(self,variant):
					variant = deepcopy(variant)
					variant['seedid'] += ray.get(iterator.next.remote())
					return collect_demonstrations(variant)
			num_workers = 10
			variant['demo_kwargs']['num_episodes'] = variant['demo_kwargs']['num_episodes']//num_workers

			samplers = [Sampler.remote() for i in range(num_workers)]
			samples = [samplers[i].sample.remote(variant) for i in range(num_workers)]
			samples = [ray.get(sample) for sample in samples]
			paths = list(sum(samples,[]))
			np.save(os.path.join("demos",variant['demo_kwargs']['save_name']), paths)
	else:
		import time
		current_time = time.time_ns()

		if args.job in ['exp']:
			variant = variants[0]
			run_variants(experiment, [variant], process_args,run_id=str(current_time))
		elif args.job in ['eval']:
			variant = variants[0]
			run_variants(eval_exp, [variant], process_args,run_id="evaluation")
		elif args.job in ['practice']:
			variant = variants[0]
			variant['seedid'] = current_time
			process_args(variant)
			collect_demonstrations(variant)
		elif args.job in ['demo']:
			variant = variants[0]
			variant['seedid'] = current_time
			process_args(variant)
			paths = collect_demonstrations(variant)
			np.save(os.path.join("demos",variant['demo_kwargs']['save_name']+str(variant['seedid'])), paths)