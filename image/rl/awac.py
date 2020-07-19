"""
AWR + SAC from demo experiment
"""

from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from railrl.launchers.experiments.awac.awac_rl import experiment, process_args

import railrl.misc.hyperparameter as hyp
from railrl.launchers.arglauncher import run_variants

from railrl.torch.sac.policies import GaussianPolicy
from railrl.torch.networks import Clamp

from env import *
from utils import *

if __name__ == "__main__":
	variant = dict(
		num_epochs=5001,
		num_eval_steps_per_epoch=1000,
		num_trains_per_train_loop=1000,
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
			output_activation=Clamp(max=0), # rewards are <= 0
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
			num_exps_per_instance=1,
			region='us-west-2',
		),

		path_loader_class=PathAdaptLoader,
		path_loader_kwargs=dict(
			obs_key="state_observation",
			demo_paths=[  # these can be loaded in awac_rl.py per env
				dict(
				    path="demos/icml2020/hand/pen_bc5.npy",
				    obs_dict=False,
				    is_demo=False,
				    train_split=0.9,
				),
			],
		),
		add_env_demos=True,
		add_env_offpolicy_data=True,
		normalize_env=False,

		load_demos=True,
		pretrain_policy=True,
		pretrain_rl=True,
	)

	env_config = deepcopy(default_config)
	s_config = deepcopy(env_config)
	f_config = deepcopy(env_config)
	l_config = deepcopy(env_config)
	r_config = deepcopy(env_config)
	ls_config = deepcopy(env_config)
	s_config.update(env_map['ScratchItch'])
	f_config.update(env_map['Feeding'])
	l_config.update(env_map['Laptop'])
	r_config.update(env_map['Reach'])
	ls_config.update(env_map['LightSwitch'])
	env_config = r_config
	env_kwargs = {**env_config,**dict(
		oracle_size=6,
		oracle='dd_target',
		num_nonnoop=10,
		input_penalty=2,
		action_type='trajectory',
	)}
	variant.update(dict(
		env_class=adapt_factory(default_class('Reach'),[window_adapt]),
		env_kwargs=env_kwargs,
	))

	search_space = {
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