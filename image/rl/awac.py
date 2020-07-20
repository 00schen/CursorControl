"""
AWR + SAC from demo experiment
"""

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.launchers.experiments.awac.awac_rl import experiment, process_args

import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

from railrl.torch.sac.policies import GaussianPolicy
from railrl.torch.networks import Clamp

from envs import *
from utils import *
import argparse
from copy import deepcopy,copy
import os

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir',default='~/share/image/rl/test',help='dir to save trials')
parser.add_argument('--env_name',)
parser.add_argument('--gpu', nargs="*", type=int, default=[0,0,1,1,2,2])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
	variant = dict(
		num_epochs=5001,
		num_eval_steps_per_epoch=200*20,
		num_trains_per_train_loop=100,
		num_expl_steps_per_train_loop=200,
		min_num_steps_before_training=1000,
		max_path_length=200,
		batch_size=1024,
		algorithm="AWAC",
		replay_buffer_size=int(1E6),

		policy_class=GaussianPolicy,
		policy_kwargs=dict(
			hidden_sizes=[256, 256, 256],
			max_log_std=0,
			min_log_std=-6,
			std_architecture="values",
		),
		qf_kwargs=dict(
			hidden_sizes=[256, 256, 256],
			# output_activation=Clamp(max=0), # rewards are <= 0
		),

		version="normal",
		collection_mode='batch',
		trainer_kwargs=dict(
			discount=0.99,
			soft_target_tau=5e-3,
			target_update_period=1,
			policy_lr=1e-3,
			qf_lr=1e-3,
			reward_scale=1,
			beta=1,
			use_automatic_entropy_tuning=False,
			alpha=0,
			compute_bc=False,
			awr_min_q=True,

			bc_num_pretrain_steps=0,
			q_num_pretrain1_steps=0,
			q_num_pretrain2_steps=25000,
			policy_weight_decay=1e-4,
			q_weight_decay=0,

			rl_weight=1.0,
			use_awr_update=True,
			use_reparam_update=False,
			reparam_weight=0.0,
			awr_weight=1.0,
			bc_weight=0.0,

			reward_transform_kwargs=None,
			terminal_transform_kwargs=dict(m=0, b=0),
		),
		launcher_config=dict(
			exp_name='test',
			mode='local',
			use_gpu=True,
		),

		path_loader_class=PathAdaptLoader,
		path_loader_kwargs=dict(
			obs_key="state_observation",
			demo_paths=[],
		),
		env_demo_path=dict(
			path=os.path.join(os.path.abspath(''),f"demos/{args.env_name}_demo.npy"),
			obs_dict=False,
			is_demo=True,
		),
		env_offpolicy_data_path=dict(
			path=os.path.join(os.path.abspath(''),f"demos/{args.env_name}_offpolicy.npy"),
			obs_dict=False,
			is_demo=False,
			train_split=0.9,
		),
		add_env_demos=True,
		add_env_offpolicy_data=True,
		normalize_env=False,

		load_demos=True,
		pretrain_policy=True,
		pretrain_rl=True,
		# parallel=len(args.gpu),
		parallel=1,
		gpus=args.gpu,
	)
	config = deepcopy(default_config)
	config.update(env_map[args.env_name])
	config.update(dict(
		oracle_size=6,
		oracle='dd_target',
		num_obs=10,
		num_nonnoop=10,
		input_penalty=2,
		action_type='trajectory',
		action_penalty=0,
	))
	variant.update(dict(
		env_class=railrl_class(args.env_name,[window_adapt]),
		env_kwargs={'config':config},
	))

	search_space = {
		'seedid': range(len(args.gpu)),
		'trainer_kwargs.beta': [0.5, ],
		'trainer_kwargs.clip_score': [0.5, ],
		'trainer_kwargs.awr_use_mle_for_vf': [True, ],
	}

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)

	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	run_variants(experiment, variants, process_args)