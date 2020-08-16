"""
AWR + SAC from demo experiment
"""

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.launchers.experiments.awac.awac_rl import experiment, process_args

import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

from railrl.torch.sac.policies import GaussianPolicy,TanhGaussianPolicy
from railrl.torch.networks import Clamp

from envs import *
# from utils import *
from rail_utils import *
import argparse
from copy import deepcopy,copy
import os

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir',default='~/share/image/rl/test',help='dir to save trials')
parser.add_argument('--env_name',)
parser.add_argument('--gpus', type=int)
parser.add_argument('--per_gpu', type=int)
parser.add_argument('--exp_name', default='a-test')
parser.add_argument('--test', type=int, default=1)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
	variant = dict(
		num_epochs=250,
		num_eval_steps_per_epoch=200*10,
		num_trains_per_train_loop=100,
		num_expl_steps_per_train_loop=200,
		min_num_steps_before_training=1000,
		max_path_length=200,
		batch_size=1024,
		algorithm="AWAC",
		replay_buffer_size=int(6e5),
		# replay_buffer_class=OnOffReplayBuffer,

		policy_class=GaussianPolicy,
		policy_kwargs=dict(
			hidden_sizes=[256]*4,
			max_log_std=0,
			min_log_std=-6,
			# std_architecture="values",
		),
		qf_kwargs=dict(
			hidden_sizes=[256]*3,
			# output_activation=Clamp(max=10), # rewards are <= 10
		),

		version="normal",
		collection_mode='batch',
		trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            # policy_lr=3E-4,
            # qf_lr=3E-4,
            reward_scale=1,
            # alpha=1.0,

            use_automatic_entropy_tuning=True,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=int(5e4),
            policy_weight_decay=1e-4,
            train_bc_on_rl_buffer=False,
            buffer_policy_sample_actions=False,

            # reparam_weight=0.0,

            awr_weight=1.0,
            bc_weight=0.0,
            compute_bc=False,
            awr_sample_actions=False,
            awr_min_q=True,
			vf_K=10,
        ),
		launcher_config=dict(
			exp_name=args.exp_name,
			mode='here_no_doodad',
			use_gpu=True,
		),
		logger_config=dict(
			snapshot_mode='last',
			# snapshot_gap=2,
			tensorboard=True,
		),

		path_loader_class=PathAdaptLoader,
		path_loader_kwargs=dict(
			obs_key="state_observation",
			demo_paths=[],
			# add_demos_to_replay_buffer=False,
		),
		add_env_demos=True,
		add_env_offpolicy_data=True,
		normalize_env=False,

		load_demos=True,
		# pretrain_policy=True,
		pretrain_rl=not args.test,
		# save_video=True,
		# image_env_kwargs=dict(
		# 	recompute_reward=False,
		# ),
		# exploration_kwargs={}
	)
	config = deepcopy(default_config)
	config.update(env_map[args.env_name])
	config.update(dict(
		oracle_size=6,
		oracle='rad_discrete_traj',
		num_obs=10,
		num_nonnoop=10,
		# threshold=.7,
		# input_penalty=.1,
		action_type='joint',
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
		'trainer_kwargs.beta': [10,30,50],
		'trainer_kwargs.reparam_weight': [0,.5,],
		'trainer_kwargs.policy_lr': [3e-4,1e-3],
		'trainer_kwargs.qf_lr': [3e-4,1e-3],
		'trainer_kwargs.q_weight_decay': [0,1e-4,3e-4],
		# 'trainer_kwargs.penalty': [.01,.1,1],
		# 'trainer_kwargs.clip_score': [100, ],
		'dense': [False,],
		# 'trainer_kwargs.awr_use_mle_for_vf': [True,],
		'env_kwargs.config.threshold': [.1,.3,.5,],
		'env_kwargs.config.input_penalty': [0,.1,.3,.5],
		# 'env_kwargs.config.target_delay': [0,30,60],
		# 'trainer_kwargs.q_num_pretrain2_steps': [int(1e4),int(2e4),int(3e4)],
	}
	dense_dicts = [
		dict(
		env_demo_path=dict(
			path=os.path.join(os.path.abspath(''),f"demos/{args.env_name}_demo3c.npy"),
			obs_dict=False,
			is_demo=True,
		),
		env_offpolicy_data_path=dict(
			path=os.path.join(os.path.abspath(''),f"demos/{args.env_name}_offpolicy3c.npy"),
			obs_dict=False,
			is_demo=False,
			train_split=0.9,
		),),
		dict(
		env_demo_path=dict(
			path=os.path.join(os.path.abspath(''),f"demos/{args.env_name}_demo1a.npy"),
			obs_dict=False,
			is_demo=True,
		),
		env_offpolicy_data_path=dict(
			path=os.path.join(os.path.abspath(''),f"demos/{args.env_name}_offpolicy1a.npy"),
			obs_dict=False,
			is_demo=False,
			train_split=0.9,
		),)
		]

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)

	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	import ray
	from ray.util import ActorPool
	from itertools import cycle,count
	ray.init(temp_dir='/tmp/ray_exp', num_gpus=args.gpus)

	@ray.remote
	class Iterators:
		def __init__(self):
			self.run_id_counter = count(0)
			# self.gpu_iter = cycle(args.gpus*args.per_gpu)
		def next(self):
			return next(self.run_id_counter)
	iterator = Iterators.options(name="global_iterator").remote()
	
	@ray.remote(num_cpus=1,num_gpus=1/args.per_gpu)
	class Runner:
		def run(self,variant):
			iterator = ray.get_actor("global_iterator")
			run_id = ray.get(iterator.next.remote())
			variant['launcher_config']['gpu_id'] = 0
			variant.update(dense_dicts[variant['dense']])
			run_variants(experiment, [variant], process_args,run_id=run_id)

	runners = [Runner.remote() for i in range(args.gpus*args.per_gpu)]
	runner_pool = ActorPool(runners)
	list(runner_pool.map(lambda a,v: a.run.remote(v), variants))