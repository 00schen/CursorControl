import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlp, MlpPolicy, ConcatMlpPolicy, ConcatRNNPolicy
from rlkit.torch.networks import Clamp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import BoltzmannPolicy, ArgmaxPolicy, EncDecQfPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.trainers import RecurEncDecCQLTrainer
from rl.replay_buffers import ModdedTrajReplayBuffer, balanced_traj_buffer_factory, pad_buffer_factory
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
	eval_config['gaze_path'] = 'bottle_gaze_data_eval1.h5'
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
		encoder = ConcatMlpPolicy(input_size=7,
							output_size=3*2,
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
		rf = th.load(file_name)['trainer/rf']
		rf.dim = -1
		gt_encoder = th.load(file_name)['trainer/encoder']
		# encoder = th.load(file_name)['trainer/encoder']
		encoder = ConcatRNNPolicy(input_size=obs_dim,
							output_size=3*2,
							hidden_sizes=[M, M],
							)
		# gt_noise = th.load(file_name)['trainer/gt_noise']
		gaze_noise = ptu.zeros(1, requires_grad=True)
		qf = th.load(file_name)['trainer/qf']
		target_qf = th.load(file_name)['trainer/target_qf']
		qf.dim = -1
		target_qf.dim = -1
		recon_decoder = ConcatRNNPolicy(input_size=3+env.observation_space.low.size,
							output_size=sum(env.feature_sizes.values()),
							hidden_sizes=[M, M],
							)
	optim_params = list(encoder.parameters())+[gaze_noise]
	if not variant['freeze_decoder']:
		optim_params += list(qf.parameters())
	if not variant['freeze_rf']:
		optim_params += list(rf.parameters())
	optimizer = optim.Adam(
			optim_params,
			lr=variant['qf_lr'],
		)

	# eval_policy = ArgmaxPolicy(
	# 	qf,
	# 	list(env.feature_sizes.keys())
	# )
	eval_policy = EncDecQfPolicy(
		encoder,
		qf,
		list(env.feature_sizes.keys())
	)
	eval_path_collector = FullPathCollector(
		eval_env,
		eval_policy,
		save_env_in_snapshot=False
	)
	if not variant['exploration_argmax']:
		expl_policy = BoltzmannPolicy(
			qf,
			logit_scale=variant['expl_kwargs']['logit_scale'])
	else:
		# expl_policy = ArgmaxPolicy(
		# 	qf,
		# 	list(env.feature_sizes.keys())
		# )
		expl_policy = EncDecQfPolicy(
			encoder,
			qf,
			list(env.feature_sizes.keys()),
			logit_scale=variant['expl_kwargs']['logit_scale']
		)
	expl_path_collector = FullPathCollector(
		env,
		expl_policy,
		save_env_in_snapshot=False,
	)
	# trainer = DDQNCQLTrainer(
	trainer = RecurEncDecCQLTrainer(
		rf=rf,
		gt_encoder=gt_encoder,
		encoder=encoder,
		recon_decoder=recon_decoder,
		# gt_logvar=gt_noise,
		gaze_logvar=gaze_noise,
		qf=qf,
		target_qf=target_qf,
		optimizer=optimizer,
		**variant['trainer_kwargs']
		)
	replay_buffer = pad_buffer_factory(ModdedTrajReplayBuffer)(
		variant['env_config']['gaze_dim'],
		variant['replay_buffer_size'],
		env=env,
		traj_len=200,
		sample_base=int(5000*200),
		env_info_sizes={'target1_reached':1},
		# target_name='target1_reached'
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
	if variant['demo_paths']:
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
	parser.add_argument('--env_name', )
	parser.add_argument('--exp_name', default='a-test')
	parser.add_argument('--no_render', action='store_false')
	parser.add_argument('--use_ray', action='store_true')
	parser.add_argument('--gpus', default=0, type=int)
	parser.add_argument('--per_gpu', default=1, type=int)
	args, _ = parser.parse_known_args()
	main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

	path_length = 200
	variant = dict(
		pretrain_path='params_ckpt.pkl',

		layer_size=256,
		exploration_argmax=True,
		expl_kwargs=dict(
			# logit_scale=100,
		),
		# replay_buffer_size=int(1e5*200),
		replay_buffer_size=int(2e6),
		trainer_kwargs=dict(
			soft_target_tau=1e-2,
			target_update_period=1,
			qf_criterion=None,
			qf_lr=5e-4,

			discount=1-(1/path_length),
			add_ood_term=-1,
			temp=1,
			min_q_weight=0,

			use_gaze_noise=True,
			global_noise=False,
		),
		algorithm_args=dict(
			batch_size=256,
			max_path_length=path_length,
			num_epochs=int(1e4),
			num_eval_steps_per_epoch=path_length,
			num_expl_steps_per_train_loop=5*path_length,
			num_train_loops_per_epoch=10,
			collect_new_paths=True,
			num_trains_per_train_loop=10,
			min_num_steps_before_training=int(1e3)
		),

		demo_paths=[
					# os.path.join(main_dir, "demos", f"bottle_debug.npy"),
					# os.path.join(main_dir, "demos", f"Bottle_model_on_policy_5000_debug1.npy"),
					# os.path.join(main_dir, "demos", f"Bottle_model_on_policy_5000_noisy.npy"),
					],

		env_config=dict(
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03, frame_skip=5, debug=False),

			action_type='disc_traj',
			smooth_alpha=.8,

			adapts=['goal','oracle','reward'],
			oracle='model',
			oracle_kwargs=dict(
				threshold=.5
			),
			gaze_dim=128,
			goal_dim=128,
			state_type=0,
			reward_max=0,
			reward_min=-1,
			reward_type='part_sparse',
			gaze_path='Bottle_gaze_data_large1.h5'
		)
	)
	search_space = {
		'seedid': [2000,2001],

		'from_pretrain': [True],
		'env_config.env_name': ['Bottle'],
		'layer_norm': [False],
		'expl_kwargs.logit_scale': [-1],

		'demo_path_proportions':[[0], ],
		'trainer_kwargs.beta': [0],
		'freeze_decoder': [False,True],
		'freeze_rf': [True,],
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
