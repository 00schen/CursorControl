from discrete_experiment import *

import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

from railrl.torch.networks import Clamp

from envs import *
from rail_utils import *
import argparse
from copy import deepcopy,copy
import os

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
	variant = dict(
		algorithm_args=dict(
			num_epochs=100,
			num_eval_steps_per_epoch=path_length*3,
			num_trains_per_train_loop=50,
			num_expl_steps_per_train_loop=path_length*3//4,
			min_num_steps_before_training=0,

			batch_size=1024,
			max_path_length=path_length,
		),

		trainer_class=DQNPavlovTrainer,

		replay_buffer_kwargs=dict(
			max_replay_buffer_size=500*path_length,
		),

		twin_q=True,
		policy_kwargs=dict(
		),
		qf_kwargs=dict(
			hidden_sizes=[512]*3,
			output_activation=Clamp(max=0), # rewards are <= 0
		),
		pf_kwargs=dict(
			hidden_size=128,
		),
		# exploration_kwargs=dict(
		# 	strategy='boltzmann'
		# ),

		version="normal",
		collection_mode='batch',
		trainer_kwargs=dict(
			# discount=0.99,
			# discount=1,
			# soft_target_tau=5e-3,
			# target_update_period=1,
			# qf_lr=3E-4,
			reward_scale=1,
		),
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

		path_loader_class=PathAdaptLoader,
		path_loader_kwargs=dict(
			obs_key="state_observation",

			demo_paths=[dict(
						path=os.path.join(os.path.abspath(''),"demos",demo),
						obs_dict=False,
						is_demo=False,
						train_split=1,
						) for demo in os.listdir(os.path.join(os.path.abspath(''),"demos")) if f"{args.env_name}_keyboardinput" in demo],
			# demo_paths=[dict(
			# 			path=os.path.join(os.path.abspath(''),"demos",f"{args.env_name}_usermodel_1001.npy"),
			# 			obs_dict=False,
			# 			is_demo=False,
			# 			train_split=1,
			# 			)],
			# add_demos_to_replay_buffer=False,
		),

		load_demos=True,
		demo_kwargs=dict(
			min_successes=100,
			min_success_rate=1,
			path_length=path_length,
			save_name=f"{args.env_name}_keyboardinput_"
			# save_name=f"{args.env_name}_usermodel_1001"
		)
	)
	from agents import *
	config = deepcopy(default_config)
	config.update(dict(
		env_name=args.env_name,
		oracle=KeyboardAgent,
		action_type='disc_traj',
		traj_len=.2,
		action_clip=.1,
		cap=0,
		step_limit=path_length,
		# elig_decay=.35,
		env_kwargs=dict(success_dist=.03,frame_skip=5),
	))
	variant.update(dict(
		env_class=railrl_class(wrapper([window_factory,cap_factory,],default_class),
					[window_adapt,cap_adapt,]),
		env_kwargs={'config':config},
	))

	search_space = {
		'seedid': [2000,],
		'env_kwargs.config.input_penalty': [1],
		'trainer_kwargs.learning_rate': [1e-3,],
		'trainer_kwargs.soft_target_tau': [5e-4,],
		'trainer_kwargs.target_update_period': [10,],
		'trainer_kwargs.discount': [.995],
		'path_loader_kwargs.add_demos_to_replay_buffer': [True],

		'replay_buffer_kwargs.window': [3],
		'replay_buffer_kwargs.decay': [.5],
		'env_kwargs.config.num_obs': [10],
		'env_kwargs.config.num_nonnoop': [10,],
		# 'exploration_kwargs.logit_scale': [int(1e3)]
	}

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)
	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	def process_args(variant):
		if not args.use_ray:
			variant['render'] = args.no_render
		if args.job in ['demo']:
			variant['env_kwargs']['config']['action_type'] = 'trajectory'
			variant['env_class'] = default_class

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
					return collect_demonstrations(variant)
			num_workers = 10
			variant['demo_kwargs']['min_successes'] = variant['demo_kwargs']['min_successes']//num_workers

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
			variant['algorithm_args']['num_eval_steps_per_epoch'] = 0
			variant['algorithm_args']['dump_tabular'] = args.no_dump_tabular
			run_variants(experiment, [variant], process_args,run_id=str(current_time))
		if args.job in ['eval']:
			variant = variants[0]
			run_variants(eval_exp, [variant], process_args,run_id="evaluation")
		elif args.job in ['demo']:
			variant = variants[0]
			variant['seedid'] = current_time
			process_args(variant)
			paths = collect_demonstrations(variant)
			np.save(os.path.join("demos",variant['demo_kwargs']['save_name']+str(variant['seedid'])), paths)
