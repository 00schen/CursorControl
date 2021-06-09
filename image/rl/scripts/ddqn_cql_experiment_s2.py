import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlpPolicy
from rlkit.torch.networks import Clamp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import EncDecQfPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.trainers import EncDecCQLTrainer1
from rl.replay_buffers import ModdedReplayBuffer
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
	eval_config['gaze_path'] = eval_config['eval_gaze_path']
	eval_env = default_overhead(variant['env_config'])
	eval_env.seed(variant['seedid']+1)

	obs_dim = env.observation_space.low.size+sum(env.feature_sizes.values())
	action_dim = env.action_space.low.size
	M = variant["layer_size"]
	upper_q = 0
	lower_q = -500

	file_name = os.path.join('util_models',variant['pretrain_path'])
	rf = th.load(file_name)['trainer/rf']
	encoder = ConcatMlpPolicy(input_size=sum(env.feature_sizes.values()),
						output_size=3,
						hidden_sizes=[M, M],
						layer_norm=variant['layer_norm'],
						hidden_activation=F.leaky_relu
						)
	# logvar = ptu.zeros(1, requires_grad=True)
	qf = th.load(file_name)['trainer/qf']
	target_qf = th.load(file_name)['trainer/qf']
	recon_decoder = ConcatMlpPolicy(input_size=3,
						output_size=encoder.input_size,
						hidden_sizes=[M, M],
						layer_norm=variant['layer_norm'],
						hidden_activation=F.leaky_relu
						)
	logvar = th.tensor(variant['logvar'])
	optim_params = list(encoder.parameters())
	if not variant['freeze_decoder']:
		optim_params += list(qf.parameters())
	# if not variant['freeze_gt']:
	# 	optim_params += list(gt_encoder.parameters())
	if not variant['freeze_rf']:
		optim_params += list(rf.parameters())
	optimizer = optim.Adam(
			optim_params,
			lr=variant['qf_lr'],
		)

	eval_policy = EncDecQfPolicy(
		qf,
		list(env.feature_sizes.keys()),
		encoder=encoder,
	)
	eval_path_collector = FullPathCollector(
		eval_env,
		eval_policy,
		save_env_in_snapshot=False
	)
	expl_policy = EncDecQfPolicy(
		qf,
		list(env.feature_sizes.keys()),
		encoder=encoder,
		logit_scale=variant['expl_kwargs']['logit_scale']
	)
	expl_path_collector = FullPathCollector(
		env,
		expl_policy,
		save_env_in_snapshot=False,
	)
	trainer = EncDecCQLTrainer1(
		rf=rf,
		encoder=encoder,
		recon_decoder=recon_decoder,
		pred_logvar=logvar,
		qf=qf,
		target_qf=target_qf,
		optimizer=optimizer,
		**variant['trainer_kwargs']
		)
	replay_buffer = ModdedReplayBuffer(
		variant['replay_buffer_size'],
		env,
		sample_base=0,
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
		pretrain_path=f'{args.env_name}_params_s1.pkl',

		layer_size=256,
		expl_kwargs=dict(
			# logit_scale=100,
		),
		replay_buffer_size=int(1e4*path_length),
		trainer_kwargs=dict(
			# soft_target_tau=1e-4,
			target_update_period=1,
			qf_criterion=None,
			qf_lr=5e-4,

			discount=1-(1/path_length),
			add_ood_term=-1,
			temp=1,
			min_q_weight=0,

			use_noise=False,
		),
		algorithm_args=dict(
			batch_size=256,
			max_path_length=path_length,
			num_epochs=1000,
			num_eval_steps_per_epoch=10*path_length,
			num_expl_steps_per_train_loop=10*path_length,
			num_train_loops_per_epoch=10,
			collect_new_paths=True,
			num_trains_per_train_loop=30,
			min_num_steps_before_training=10
		),

		env_config=dict(
			env_name=args.env_name,
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03, frame_skip=5, debug=False),

			action_type='disc_traj',
			smooth_alpha=.8,

			factories=['session'],
			adapts=['goal','static_gaze','reward'],
			gaze_dim=128,
			state_type=0,
			reward_max=0,
			reward_min=-1,
			reward_type='sparse',
			gaze_path=f'{args.env_name}_gaze_data_train.h5',
			eval_gaze_path=f'{args.env_name}_gaze_data_eval.h5'
			# gaze_path='Bottle_gaze_data_large1.h5',
			# eval_gaze_path='bottle_gaze_data_eval1.h5'
			# gaze_path='switch_gaze_data_train.h5',
			# eval_gaze_path='switch_gaze_data_eval.h5'
		)
	)
	search_space = {
		'seedid': [2000,2001],

		'from_pretrain': [True],
		'layer_norm': [True],
		'expl_kwargs.logit_scale': [10],
		'algorithm_args.num_train_loops_per_epoch': [10,100],
		'trainer_kwargs.soft_target_tau': [1e-3,1e-4],
		'logvar': [-10],

		'freeze_decoder': [False,],
		'freeze_rf': [True],
		'trainer_kwargs.use_supervised': ['target']
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
