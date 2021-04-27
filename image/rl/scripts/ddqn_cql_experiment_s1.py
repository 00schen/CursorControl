import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlpPolicy
from rlkit.torch.networks import Clamp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import EncDecPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.trainers import EncDecCQLTrainer
from rl.replay_buffers import HERReplayBuffer, ModdedReplayBuffer, balanced_buffer_factory
from rl.scripts.run_util import run_exp

import os
from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
from torch.nn import functional as F
from torch import optim


def experiment(variant):
	from rlkit.core import logger
	import torch as th

	env = default_overhead(variant['env_config'])
	env.seed(variant['seedid'])
	eval_config = variant['env_config'].copy()
	eval_env = default_overhead(variant['env_config'])
	eval_env.seed(variant['seedid']+1)

	obs_dim = env.observation_space.low.size+sum(env.feature_sizes.values())
	action_dim = env.action_space.low.size
	M = variant["layer_size"]
	upper_q = 0
	lower_q = -500

	if not variant['from_pretrain']:
		rf = ConcatMlpPolicy(input_size=10*2,
							output_size=1,
							hidden_sizes=[M, M],
							layer_norm=variant['layer_norm'],
							hidden_activation=F.leaky_relu
							)
		# gt_noise = ptu.zeros(1, requires_grad=True)
		gaze_noise = ptu.zeros(1, requires_grad=True)
		qf = ConcatMlpPolicy(input_size=10,
							output_size=action_dim,
							hidden_sizes=[M, M, M],
							layer_norm=variant['layer_norm'],
							output_activation=Clamp(max=upper_q, min=lower_q),
							hidden_activation=F.leaky_relu
							)
		target_qf = ConcatMlpPolicy(input_size=10,
							output_size=action_dim,
							hidden_sizes=[M, M, M],
							layer_norm=variant['layer_norm'],
							output_activation=Clamp(max=upper_q, min=lower_q),
							hidden_activation=F.leaky_relu
							)
	else:
		file_name = os.path.join('util_models',variant['pretrain_path'])
		rf = ConcatMlpPolicy(input_size=obs_dim*2,
							output_size=1,
							hidden_sizes=[M, M],
							layer_norm=variant['layer_norm'],
							hidden_activation=F.leaky_relu
							)
		logvar = ptu.zeros(1, requires_grad=True)
		qf = th.load(file_name)['trainer/qf']
		target_qf = th.load(file_name)['trainer/qf']
	
	optimizer = optim.Adam(
		list(rf.parameters())+list(qf.parameters())+[logvar],
		lr=variant['qf_lr'],
	)

	eval_policy = EncDecPolicy(
		qf,
		list(env.feature_sizes.keys())
	)
	eval_path_collector = FullPathCollector(
		eval_env,
		eval_policy,
		save_env_in_snapshot=False
	)
	expl_policy = EncDecPolicy(
		qf,
		list(env.feature_sizes.keys()),
		logit_scale=variant['expl_kwargs']['logit_scale']
	)
	expl_path_collector = FullPathCollector(
		env,
		expl_policy,
		save_env_in_snapshot=False,
	)
	trainer = EncDecCQLTrainer(
		rf=rf,
		gt_logvar=logvar,
		qf=qf,
		target_qf=target_qf,
		optimizer=optimizer,
		**variant['trainer_kwargs']
		)
	replay_buffer = variant['buffer_type'](
		variant['replay_buffer_size'],
		env,
		# target_name='target1_reached',
		# env_info_sizes={'target1_reached': 1},
		sample_base=int(5000*200),
	)
	algorithm = TorchBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=env,
		evaluation_env=eval_env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		**variant['algorithm_args']
	)
	algorithm.to(ptu.device)
	path_loader = SimplePathLoader(
		demo_path=variant['demo_paths'],
		demo_path_proportion=variant['demo_path_proportions'],
		replay_buffer=replay_buffer,
	)
	path_loader.load_demos()
	from rlkit.core import logger

	if variant.get('render', False):
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
	main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

	path_length = 200
	variant = dict(
		pretrain_path=f'{args.env_name}_params.pkl',

		layer_size=256,
		expl_kwargs=dict(
			# logit_scale=100,
		),
		# replay_buffer_size=int(4e6),
		trainer_kwargs=dict(
			# soft_target_tau=1e-4,
			target_update_period=1,
			qf_criterion=None,
			qf_lr=5e-4,

			discount=1-(1/path_length),
			add_ood_term=-1,
			temp=1,
			min_q_weight=0,

			use_noise=True,
		),
		algorithm_args=dict(
			batch_size=256,
			max_path_length=path_length,
			num_epochs=int(1e6),
			num_eval_steps_per_epoch=5*path_length,
			num_expl_steps_per_train_loop=path_length,
			num_train_loops_per_epoch=100,
			collect_new_paths=True,
			num_trains_per_train_loop=100,
			min_num_steps_before_training=int(1e3)
		),

		demo_paths=[
					os.path.join(main_dir, "demos", f"{args.env_name}_model_on_policy_5000_debug.npy"),
					],

		env_config=dict(
			env_name=args.env_name,
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03, frame_skip=5, debug=False),

			action_type='disc_traj',
			smooth_alpha=.8,

			adapts=['goal','reward'],
			gaze_dim=128,
			state_type=0,
			reward_max=0,
			reward_min=-1,
			reward_type='sparse',
		)
	)
	search_space = {
		'seedid': [2000,2001],

		'from_pretrain': [True],
		'layer_norm': [False,],
		'expl_kwargs.logit_scale': [-1,10],
		'trainer_kwargs.soft_target_tau': [1e-2],

		'demo_path_proportions':[[int(5e3)], ],
		'trainer_kwargs.beta': [.001],
		'buffer_type': [ModdedReplayBuffer],
		# 'buffer_type': [HERReplayBuffer],
		'replay_buffer_size': [int(2e6)],
	}

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)
	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	def process_args(variant):
		variant['trainer_kwargs']['learning_rate'] = variant['trainer_kwargs'].pop('qf_lr')
		variant['qf_lr'] = variant['trainer_kwargs']['learning_rate']
		variant['env_config']['seedid'] = variant['seedid']
		if not args.use_ray:
			variant['render'] = args.no_render
	args.process_args = process_args

	run_exp(experiment, variants, args)
