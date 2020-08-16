"""
AWR + SAC from demo experiment
"""

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
# from railrl.launchers.experiments.awac.awac_rl import process_args
from discrete_experiment import *

import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

from railrl.torch.networks import Clamp

from envs import *
from rail_utils import *
# from dqfd import *
import argparse
from copy import deepcopy,copy
import os

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir',default='~/share/image/rl/test',help='dir to save trials')
parser.add_argument('--env_name',)
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument('--gpus', type=int)
parser.add_argument('--per_gpu', type=int)
parser.add_argument('--exp_name', default='a-test')
parser.add_argument('--test', type=int, default=1)
args, _ = parser.parse_known_args()

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
	variant = dict(
		num_epochs=50,
		# num_eval_steps_per_epoch=200*10,
		num_eval_steps_per_epoch=0,
		num_trains_per_train_loop=100,
		num_expl_steps_per_train_loop=200,
		min_num_steps_before_training=0,
		max_path_length=200,
		batch_size=5,
		# replay_buffer_size=int(5e4),
		# replay_buffer_class=DQfDReplayBuffer,
		# trainer_class=DQfDTrainer,

		# replay_buffer_class=BalancedReplayBuffer,
		trainer_class=DQNPavlovTrainer,

		replay_buffer_kwargs=dict(
			max_num_traj=250,
			traj_max=200,
			subtraj_len=100,
		),

		twin_q=True,
		policy_kwargs=dict(
		),
		qf_kwargs=dict(
			hidden_sizes=[256]*3,
			# output_activation=Clamp(max=10), # rewards are <= 10
		),
		pf_kwargs=dict(
			hidden_size=128,
		),

		version="normal",
		collection_mode='batch',
		trainer_kwargs=dict(
            discount=0.99,
            # soft_target_tau=5e-3,
            target_update_period=1,
            # policy_lr=3E-4,
            # qf_lr=3E-4,
            reward_scale=1,
            # alpha=1.0,
			pretrain_steps=int(5e4),
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
			demo_paths=[
				dict(
					path=os.path.join(os.path.abspath(''),f"demos/LightSwitch_user_10{0 if i < 10 else ''}{i}.npy"),
					obs_dict=False,
					is_demo=False,
					train_split=0.9,
				)
				for i in range(12)
			],
			# add_demos_to_replay_buffer=False,
		),
		# add_env_demos=True,
		# add_env_offpolicy_data=True,
		normalize_env=False,

		exploration_kwargs=dict(
			strategy='boltzmann'
		),

		load_demos=True,
		pretrain_rl=not args.test,
	)
	config = deepcopy(default_config)
	config.update(env_map[args.env_name])
	config.update(dict(
		oracle_size=6,
		# oracle='user_model',
		oracle='user',
		num_obs=10,
		num_nonnoop=10,
		# threshold=.7,
		# input_penalty=.1,
		input_penalty=0,
		action_type='cat_target',
		action_penalty=0,
		include_target=True,
		# target_delay=80,
	))
	variant.update(dict(
		# env_class=new_target_factory(railrl_class(args.env_name,[window_adapt])),
		env_class=railrl_class(args.env_name,[window_adapt]),
		env_kwargs={'config':config},
	))

	search_space = {
		'seedid': [2000,],
		'dense': [True,],
		# 'env_kwargs.config.threshold': [.1,.3,.5,],
		'trainer_kwargs.learning_rate': [3e-4,1e-3,3e-3],
		'trainer_kwargs.soft_target_tau': [3e-4,1e-3,3e-3],
		'policy_kwargs.penalty': [0,.1,.5,1],
		'policy_kwargs.q_coeff': [0,1],
		# 'env_kwargs.config.input_penalty': [0,.1,.3,.5],
		# 'exploration_kwargs.epsilon': [.3,.4,.5]
		'exploration_kwargs.logit_scale': [10,100,1000]
	}
	# search_space = {
	# 	'seedid': [45,],
	# 	'dense': [True,],
	# 	# 'env_kwargs.config.threshold': [.1,.3,.5,],
	# 	'trainer_kwargs.learning_rate': [3e-4,],
	# 	'trainer_kwargs.soft_target_tau': [3e-4,],
	# 	'policy_kwargs.penalty': [.5,],
	# 	'policy_kwargs.q_coeff': [1,],
	# 	# 'env_kwargs.config.input_penalty': [0,.1,.3,.5],
	# 	'exploration_kwargs.logit_scale': [1000,]
	# }

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)

	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	import ray
	from ray.util import ActorPool
	from itertools import cycle,count
	ray.init(temp_dir='/tmp/ray_exp', num_gpus=args.gpus if args.use_gpu else 0)

	@ray.remote
	class Iterators:
		def __init__(self):
			self.run_id_counter = count(0)
			# self.gpu_iter = cycle(args.gpus*args.per_gpu)
		def next(self):
			return next(self.run_id_counter)
	iterator = Iterators.options(name="global_iterator").remote()
	
	@ray.remote(num_cpus=1,num_gpus=1/args.per_gpu if args.use_gpu else 0)
	class Runner:
		def run(self,variant):
			iterator = ray.get_actor("global_iterator")
			run_id = ray.get(iterator.next.remote())
			variant['launcher_config']['gpu_id'] = 0
			# variant.update(dense_dicts[variant['dense']])
			run_variants(experiment, [variant], process_args,run_id=run_id)

	runners = [Runner.remote() for i in range(args.gpus*args.per_gpu)]
	runner_pool = ActorPool(runners)
	list(runner_pool.map(lambda a,v: a.run.remote(v), variants))

	# for variant in variants:
	# 	run_variants(experiment, [variant], process_args,run_id="run_0")