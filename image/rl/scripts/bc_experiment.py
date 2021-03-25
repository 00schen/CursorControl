import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import VAEGazePolicy, MlpPolicy, TransferEncoderPolicy, Mlp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer

from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.misc.balanced_replay_buffer import BalancedReplayBuffer,CycleGANReplayBuffer
from rl.trainers import DiscreteVAEBCTrainerTorch, DiscreteBCTrainerTorch, DiscreteCycleGANBCTrainerTorch
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rl.path_collectors import FullPathCollector
from rl.policies import ArgmaxPolicy
from rl.scripts.run_util import run_exp

import os
import torch
import gtimer as gt
from pathlib import Path
import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import setup_logger, reset_execution_environment
import argparse

from torch import optim


def experiment(variant):
	from rlkit.core import logger
	gaze_dim = 128
	# embedding_dim = variant['embedding_dim']
	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid'])
	M = variant["layer_size"]
	obs_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size
	# policy = VAEGazePolicy(input_size=obs_dim,
	# 					 output_size=action_dim,
	# 					 decoder_hidden_sizes=[M, M, M, M],
	# 					 layer_norm=True,
	# 					 gaze_dim=50,
	# 					 embedding_dim=embedding_dim,
	# 					 )
	
	# if variant['use_pretrained_decoder']:
	# 	decoder = torch.load(variant['decoder_path'], map_location='cpu')['trainer/policy']
	# 	policy = TransferEncoderPolicy(decoder,pred_dim=3,decoder_pred_dim=50,
	# 									layer_norm=variant['layer_norm'])
	# 	policy_optimizer = optim.Adam(
	# 		# list(policy.encoder.parameters())+list(policy.layer_norm.parameters()),
	# 		policy.encoder.parameters(),
	# 		lr=5e-4,
	# 	)
	# else:
	# 	policy = MlpPolicy(input_size=obs_dim,
	# 					 output_size=action_dim,
	# 					 hidden_sizes=[M, M, M, M],
	# 					 layer_norm=variant['layer_norm'],
	# 					 )
	# 	policy_optimizer = optim.Adam(
	# 		policy.parameters(),
	# 		lr=5e-4,
	# 	)
	# policy.to(ptu.device)

	policy_encoder = Mlp(input_size=gaze_dim,
                           output_size=gaze_dim,
                           hidden_sizes=(64,64)
                           )
	policy_decoder = MlpPolicy(input_size=obs_dim,
						 output_size=action_dim,
						 hidden_sizes=[M, M, M, M, M],
						 layer_norm=variant['layer_norm'],
						 )
	policy = TransferEncoderPolicy(policy_encoder,policy_decoder)
	policy_optimizer = optim.Adam(
		policy.parameters(),
		lr=5e-4,
	)

	trainer_type = DiscreteCycleGANBCTrainerTorch
	trainer = trainer_type(
		policy=policy,
		optimizer=policy_optimizer,
		encoder=policy_encoder,
		gan_kwargs=variant['gan_kwargs']
		# policy_lr=variant['trainer_kwargs']['lr']
	)

	replay_buffer = CycleGANReplayBuffer(
		variant['replay_buffer_size'],
		env,
		# target_name=variant['balance_feature'],
		# env_info_sizes={variant['balance_feature']:1},
		target_name='gaze',
		# false_prop=variant['false_prop'],
		env_info_sizes={'task_success':1, 'target1_reached':1, 'gaze':1},
	)

	path_loader = SimplePathLoader(
		demo_path=variant['demo_paths'],
		demo_path_proportion=variant['demo_path_proportions'],
		replay_buffer=replay_buffer,
	)
	path_loader.load_demos()

	eval_policy = ArgmaxPolicy(
		policy
	)

	eval_path_collector = FullPathCollector(
		env,
		eval_policy,
		save_env_in_snapshot=False
	)

	expl_policy = eval_policy

	expl_path_collector = FullPathCollector(
		env,
		expl_policy,
		save_env_in_snapshot=False
	)

	# logger.remove_tabular_output(
	#     'progress.csv', relative_to_snapshot_dir=True,
	# )
	# logger.add_tabular_output(
	#     'pretrain.csv', relative_to_snapshot_dir=True,
	# )

	bc_algorithm = TorchBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=env,
		evaluation_env=env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		**variant['bc_args']
	)

	bc_algorithm.to(ptu.device)
	bc_algorithm.train()
	gt.reset_root()
	# logger.remove_tabular_output(
	#     'pretrain.csv', relative_to_snapshot_dir=True,
	# )
	# logger.add_tabular_output(
	#     'progress.csv', relative_to_snapshot_dir=True,
	# )
	# pretrain_file_path = os.path.join(logger.get_snapshot_dir(), 'pretrain.pkl')
	# torch.save(policy, pretrain_file_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', )
	parser.add_argument('--exp_name', default='bc_gaze')
	parser.add_argument('--no_render', action='store_false')
	parser.add_argument('--use_ray', action='store_true')
	parser.add_argument('--gpus', default=0, type=int)
	parser.add_argument('--per_gpu', default=1, type=int)
	args, _ = parser.parse_known_args()
	main_dir = str(Path(__file__).resolve().parents[2])
	print(main_dir)

	path_length = 200
	num_epochs = int(1e4)
	variant = dict(
		decoder_path=os.path.join(main_dir, 'util_models', 'bc_transfer_1.pkl'),
		layer_size=128,
		replay_buffer_size=int(1e6),
		trainer_kwargs=dict(
			# lr=5e-4,
		),
		bc_args=dict(
			batch_size=128,
			max_path_length=path_length,
			num_epochs=500,
			num_eval_steps_per_epoch=path_length * 3,
			num_expl_steps_per_train_loop=0,
			collect_new_paths=False,
			# eval_paths=False,
			num_trains_per_train_loop=1000,
		),
		# demo_paths=[
		# 	os.path.join(main_dir, "demos",'Bottle_model_on_policy_100_bottle_two_target_static.npy'),
		# 	# os.path.join(main_dir, "demos",'Bottle_model_on_policy_1000_bottle_two_target.npy')
		# ],
		gan_kwargs=dict(),

		env_kwargs={'config': dict(
			env_name=args.env_name,
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03, frame_skip=5, stochastic=True),
			# oracle='sim_gaze_model',
			oracle='dummy_gaze',
			oracle_kwargs=dict(),
			# gaze_oracle_kwargs={'mode': 'rl'},
			action_type='disc_traj',
			smooth_alpha=.8,

			# adapts=['high_dim_user'],
			# input_in_obs=False,
			# apply_projection=False,
			# state_type=0,
			adapts=['static_gaze'],
			gaze_dim=128,
		)},

		# seedid=2000,
		# demo_path_proportions=[1000],
		balance_feature='target1_reached',
	)

	search_space = {
		'seedid':[2000,2001,2002,2003],
		# use_pretrained_decoder=[True,False],
		'layer_norm':[False],
		# demo_path_proportions=[[1000,100],[1000,50],[1000,20],[1000,10]],
		'demo_path_proportions':[[int(1e3),int(1000)], ],
		'demo_paths':[[
					# os.path.join(main_dir, "demos", f"Bottle_model_on_policy_1000_bottle_two_target.npy"),
					# os.path.join(main_dir, "demos", f"Bottle_model_on_policy_100_bottle_two_target_static_gaze.npy"),
					os.path.join(main_dir, "demos", f"Bottle_model_on_policy_100_debug.npy"),
					os.path.join(main_dir, "demos", f"Bottle_model_on_policy_100_debug_gaze.npy"),
					]],
		'gan_kwargs.gaze_recon_w': [0],
		'gan_kwargs.target_pos_recon_w': [0],
		'gan_kwargs.adversarial_w': [0],
	}
	# search_space = dict(
	# 	seedid=[2000],
	# 	use_pretrained_decoder=[False],
	# 	layer_norm=[False],
	# 	demo_path_proportions=[[1000]],
	# )

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)
	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)


	def process_args(variant):
		variant['env_kwargs']['config']['seedid'] = variant['seedid']

	args.main_dir = main_dir
	args.process_args = process_args
	run_exp(experiment, variants, args)

