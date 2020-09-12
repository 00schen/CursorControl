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
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--use_ray', type=int, default=1)
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument('--gpus', type=int)
parser.add_argument('--per_gpu', type=int)
args, _ = parser.parse_known_args()

from torch import sigmoid
from torch import nn
class Sigmoid(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.__name__ = "Sigmoid"

    def forward(self, x):
        return sigmoid(x, **self.kwargs)

def process_args(variant):
    if variant.get("debug", False):
        variant['max_path_length'] = 50
        variant['batch_size'] = 5
        variant['num_epochs'] = 5
        variant['num_eval_steps_per_epoch'] = 100
        variant['num_expl_steps_per_train_loop'] = 100
        variant['num_trains_per_train_loop'] = 10
        variant['min_num_steps_before_training'] = 100
        variant['trainer_kwargs']['bc_num_pretrain_steps'] = min(10, variant['trainer_kwargs'].get('bc_num_pretrain_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain1_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain1_steps', 0))
        variant['trainer_kwargs']['q_num_pretrain2_steps'] = min(10, variant['trainer_kwargs'].get('q_num_pretrain2_steps', 0))

if __name__ == "__main__":
	path_length = 200
	variant = dict(
		algorithm_args=dict(
			num_epochs=2000,
			# num_eval_steps_per_epoch=2*path_length,
			num_eval_steps_per_epoch=path_length,
			num_trains_per_train_loop=10,
			num_expl_steps_per_train_loop=path_length,
			min_num_steps_before_training=0,
			pf_train_frequency=1,

			traj_batch_size=10,	
			batch_size=1024,
			max_path_length=path_length//3,
		),

		trainer_class=DQNPavlovTrainer,

		replay_buffer_kwargs=dict(
			max_num_traj=1000,
			traj_max=path_length,
			subtraj_len=path_length,
		),

		twin_q=True,
		policy_kwargs=dict(
		),
		qf_kwargs=dict(
			hidden_sizes=[512]*3,
			# output_activation=Sigmoid(),
			# output_activation=Clamp(max=0), # rewards are <= 0
		),
		pf_kwargs=dict(
			hidden_size=128,
		),

		version="normal",
		collection_mode='batch',
		trainer_kwargs=dict(
            # discount=0.99,
			# discount=1,
            # soft_target_tau=5e-3,
            # target_update_period=1,
            # policy_lr=3E-4,
            # qf_lr=3E-4,
            reward_scale=1,
            # alpha=1.0,
        ),
		launcher_config=dict(
			exp_name=args.exp_name,
			mode='here_no_doodad',
			use_gpu=args.use_gpu,
		),
		logger_config=dict(
			snapshot_mode='last',
			# snapshot_gap=2,
			tensorboard=True,
		),

		path_loader_class=PathAdaptLoader,
		path_loader_kwargs=dict(
			obs_key="state_observation",
			# demo_paths=[
			# 	dict(
			# 		path=os.path.join(os.path.abspath(''),f"demos/LightSwitch_user_10{0 if i < 10 else ''}{i}.npy"),
			# 		obs_dict=False,
			# 		is_demo=False,
			# 		train_split=0.9,
			# 	)
			# 	for i in range(12)
			# # ],
			# ]+[
			# 	dict(
			# 		path=os.path.join(os.path.abspath(''),f"demos/LightSwitch_user1_10{0 if i < 10 else ''}{i}.npy"),
			# 		obs_dict=False,
			# 		is_demo=False,
			# 		train_split=0.9,
			# 	)
			# 	for i in range(5)
			# ],
			demo_paths=[dict(
					path=os.path.join(os.path.abspath(''),f"demos/{args.env_name}_usermodel_1000.npy"),
					obs_dict=False,
					is_demo=False,
					train_split=0.9,
				)],
			# add_demos_to_replay_buffer=False,
		),
		# add_env_demos=True,
		# add_env_offpolicy_data=True,

		exploration_kwargs=dict(
			strategy='boltzmann'
		),

		load_demos=True,
		pretrain_rl=args.pretrain,
		save_path = os.path.join('logs',args.exp_name,'run14','id0',),

		demo_kwargs=dict(
			total_paths_per_target=100,
			fails_per_success=1,
			path_length=path_length,
			save_name=f"{args.env_name}_usermodel_1000"
		)
	)
	from agents import *
	config = deepcopy(default_config)
	config.update(dict(
		env_name=args.env_name,
		oracle=UserModelAgent,
		num_obs=5,
		num_nonnoop=5,
		action_type='disc_traj',
		cap=0,
		step_limit=path_length,
		action_clip=.1,
		env_kwargs=dict(success_dist=.05),
	))
	wrapper_class = railrl_class(wrapper([window_factory,target_factory,metric_factory],default_class),[cap_adapt,window_adapt,target_adapt,])
	variant.update(dict(
		env_class=wrapper_class if args.job != 'demo' else default_class,
		env_kwargs={'config':config},
	))

	search_space = {
		'seedid': [2000,2001],
		'env_kwargs.config.input_penalty': [1],
		# 'trainer_kwargs.learning_rate': [1e-4,3e-4,1e-3],
		# 'trainer_kwargs.soft_target_tau': [1e-3,3e-3,5e-3,],
		'trainer_kwargs.learning_rate': [1e-4,],
		'trainer_kwargs.soft_target_tau': [1e-3,],
		'algorithm_args.num_pf_trains_per_train_loop': [10000,],
		'pf_kwargs.num_layers': [1],
		'trainer_kwargs.discount': [.99,],
		'trainer_kwargs.pf_lr': [3e-4,],
		'trainer_kwargs.pf_decay': [0,],
		'trainer_kwargs.target_update_period': [10,],

		'exploration_kwargs.logit_scale': [100]
	}

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)

	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	if args.use_ray:
		import ray
		from ray.util import ActorPool
		from itertools import cycle,count
		ray.init(temp_dir='/tmp/ray_exp', num_gpus=args.gpus if args.use_gpu else 0)

		@ray.remote
		class Iterators:
			def __init__(self):
				self.run_id_counter = count(0)
			def next(self):
				return next(self.run_id_counter)
		iterator = Iterators.options(name="global_iterator").remote()
		
		if args.job in ['exp','eval']:
			@ray.remote(num_cpus=1,num_gpus=1/args.per_gpu if args.use_gpu else 0)
			class Runner:
				def run(self,variant):
					iterator = ray.get_actor("global_iterator")
					run_id = ray.get(iterator.next.remote())
					variant['launcher_config']['gpu_id'] = 0
					# run_variants(experiment, [variant], process_args,run_id=run_id)
					run_variants(pf_exp, [variant], process_args,run_id=run_id)
					# run_variants(eval_exp, [variant], process_args,run_id="evaluation")
			runners = [Runner.remote() for i in range(args.gpus*args.per_gpu)]
			runner_pool = ActorPool(runners)
			list(runner_pool.map(lambda a,v: a.run.remote(v), variants))
		elif args.job in ['demo']:
			variant = variants[0]
			print("function called")
			@ray.remote(num_cpus=1,num_gpus=0)
			class Sampler:
				def sample(self,variant):
					return collect_demonstrations(variant)
			num_workers = 10
			variant['demo_kwargs']['paths_per_target'] = variant['demo_kwargs']['total_paths_per_target']//num_workers

			samplers = [Sampler.remote() for i in range(num_workers)]
			samples = [samplers[i].sample.remote(variant) for i in range(num_workers)]
			samples = [ray.get(sample) for sample in samples]

			paths = list(sum(samples,[]))

			np.save(os.path.join("demos",variant['demo_kwargs']['save_name']), paths)
	else:
		if args.job in ['exp','eval']:
			for variant in variants:
				run_variants(eval_exp, [variant], process_args,run_id="evaluation")
				# run_variants(experiment, [variant], process_args,run_id="run_0")
		elif args.job in ['demo']:
			variant = variants[0]
			variant['render'] = True
			variant['demo_kwargs']['paths_per_target'] = variant['demo_kwargs']['total_paths_per_target']
			collect_demonstrations(variant)