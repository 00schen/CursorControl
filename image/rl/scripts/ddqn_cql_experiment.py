import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp, ConcatMlp, MlpPolicy
from rlkit.torch.networks import Clamp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import BoltzmannPolicy, OverridePolicy, ComparisonMergePolicy, ArgmaxPolicy
from rl.path_collectors import FullPathCollector, CustomPathCollector, rollout
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.trainers import DDQNCQLTrainer
from rl.misc.balanced_replay_buffer import BalancedReplayBuffer, GazeReplayBuffer
from rl.scripts.run_util import run_exp
from rlkit.torch.networks import VAEGazePolicy, TransferEncoderPolicy

import os
from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
from torch.nn import functional as F
from torch import optim


def experiment(variant):
	from rlkit.core import logger
	import torch as th

	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid'])
	eval_config = variant['env_kwargs']['config'].copy()
	eval_config['gaze_path'] = 'bottle_gaze_data_eval1.h5'
	eval_env = default_overhead(variant['env_kwargs']['config'])
	eval_env.seed(variant['seedid']+1)

	obs_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size
	M = variant["layer_size"]
	upper_q = 10
	lower_q = -500

	if not variant['from_pretrain']:
		qf = MlpPolicy(input_size=obs_dim,
							output_size=action_dim,
							hidden_sizes=[M, M, M, M, M],
							layer_norm=variant['layer_norm'],
							output_activation=Clamp(max=upper_q, min=lower_q),
							)
		target_qf = MlpPolicy(input_size=obs_dim,
							output_size=action_dim,
							hidden_sizes=[M, M, M, M, M],
							layer_norm=variant['layer_norm'],
							output_activation=Clamp(max=upper_q, min=lower_q),
							)
	else:
		file_name = os.path.join('util_models',variant['pretrain_path'])
		qf = th.load(file_name)['trainer/qf']
		target_qf = th.load(file_name)['trainer/target_qf']
	optimizer = optim.Adam(
		qf.parameters(),
		lr=variant['qf_lr'],
	)

	eval_policy = ArgmaxPolicy(
		qf,
	)
	# eval_path_collector = CustomPathCollector(
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
		expl_policy = ArgmaxPolicy(
			qf,
		)
	if variant['exploration_strategy'] == 'merge_arg':
		expl_policy = ComparisonMergePolicy(env.rng, expl_policy, env.oracle.size)
	elif variant['exploration_strategy'] == 'override':
		expl_policy = OverridePolicy(expl_policy, env, env.oracle.size)
	expl_path_collector = FullPathCollector(
		env,
		expl_policy,
		save_env_in_snapshot=False,
	)
	trainer = DDQNCQLTrainer(
		qf=qf,
		target_qf=target_qf,
		optimizer=optimizer,
		**variant['trainer_kwargs']
		)
	replay_buffer = GazeReplayBuffer(
		variant['env_kwargs']['config']['gaze_path'],
		variant['replay_buffer_size'],
		env,
		target_name='target1_reached',
		# false_prop=variant['false_prop'],
		env_info_sizes={'task_success':1, 'target1_reached':1},
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
		# from_pretrain=True,
		pretrain_path='cql_end_to_end.pkl',
		layer_size=128,
		exploration_argmax=True,
		exploration_strategy='',
		expl_kwargs=dict(
			logit_scale=1000,
		),
		replay_buffer_size=int(2e4)*path_length,
		trainer_kwargs=dict(
			soft_target_tau=1e-2,
			target_update_period=1,
			qf_criterion=None,

			discount=1-(1/path_length),
			add_ood_term=-1,
		),
		algorithm_args=dict(
			batch_size=256,
			max_path_length=path_length,
			num_epochs=int(3e4),
			num_eval_steps_per_epoch=path_length,
			num_expl_steps_per_train_loop=path_length,
			collect_new_paths=False,
			num_trains_per_train_loop=1000,
		),

		load_demos=True,
		demo_paths=[
					os.path.join(main_dir, "demos", f"Bottle_model_on_policy_5000_debug.npy"),
					],
		pretrain=False,

		env_kwargs={'config':dict(
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03, frame_skip=5,),

			oracle='dummy_gaze',
			oracle_kwargs=dict(
				threshold=.5
			),
			action_type='disc_traj',
			smooth_alpha=.8,

			adapts=['static_gaze', 'reward'],
			gaze_dim=128,
			apply_projection=False,
			state_type=0,
			reward_max=0,
			reward_min=-1,
			input_penalty=1,
			reward_type='part_sparse',
			gaze_path='Bottle_gaze_data_large1.h5'
		)}
	)
	search_space = {
		'seedid': [2000,2001,2002],

		'trainer_kwargs.temp': [1],
		'from_pretrain': [False],
		'trainer_kwargs.min_q_weight': [20,10,5],
		'env_kwargs.config.env_name': ['Bottle'],
		'layer_norm': [True],

		'demo_path_proportions':[[int(5e3),], ],
		'trainer_kwargs.qf_lr': [1e-5],
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
		variant['env_kwargs']['config']['seedid'] = variant['seedid']
		if not args.use_ray:
			variant['render'] = args.no_render
	args.process_args = process_args

	run_exp(experiment, variants, args)
