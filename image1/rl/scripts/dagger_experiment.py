import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp, ConcatMlp
from rlkit.torch.networks import Clamp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import BoltzmannPolicy, OverridePolicy, ComparisonMergePolicy, ArgmaxPolicy
from rl.path_collectors import FullPathCollector, CustomPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.trainers import DisBCTrainer
from rl.misc.balanced_replay_buffer import BalancedReplayBuffer
from rl.scripts.run_util import run_exp
from rl.misc.utils import make_alpha_relu

import os
from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
from torch.nn import functional as F


def experiment(variant):
	from rlkit.core import logger

	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid'])

	obs_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size
	M = variant["layer_size"]
	qf1 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_sizes=[M, M, M, M],
		# hidden_activation=make_alpha_relu(variant['dropout_p']),
		hidden_activation=F.leaky_relu,
		layer_norm=variant['layer_norm'],
		output_activation=Clamp(min=-10, max=10),
	)
	eval_policy = ArgmaxPolicy(
		qf1,
	)
	# eval_path_collector = CustomPathCollector(
	eval_path_collector = FullPathCollector(
		env,
		eval_policy,
		save_env_in_snapshot=False,
	)
	if not variant['exploration_argmax']:
		expl_policy = BoltzmannPolicy(
			qf1,
			logit_scale=variant['expl_kwargs']['logit_scale'])
	else:
		expl_policy = ArgmaxPolicy(
			qf1,
		)
	if variant['exploration_strategy'] == 'merge_arg':
		expl_policy = ComparisonMergePolicy(env.rng, expl_policy, env.oracle.size)
	elif variant['exploration_strategy'] == 'override':
		expl_policy = OverridePolicy(expl_policy, env, env.oracle.size)
	expl_path_collector = FullPathCollector(
		env,
		expl_policy,
		save_env_in_snapshot=False
	)
	trainer = DisBCTrainer(
		policy=qf1,
		**variant['trainer_kwargs']
	)
	replay_buffer = BalancedReplayBuffer(
		variant['replay_buffer_size'],
		env,
		target_name='noop',
		false_prop=variant['false_prop'],
	)
	algorithm = TorchBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=env,
		evaluation_env=env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		**variant['algorithm_args']
	)
	algorithm.to(ptu.device)
	if variant.get('load_demos', False):
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

	path_length = 400
	num_epochs = int(5000)
	variant = dict(
		layer_size=128,
		exploration_argmax=True,
		exploration_strategy='override',
		expl_kwargs=dict(
			logit_scale=1000,
		),
		replay_buffer_size=int(2e4)*path_length,
		# intervention_prop=.5
		trainer_kwargs=dict(
		),
		algorithm_args=dict(
			batch_size=256,
			max_path_length=path_length,
			eval_path_length=400,
			num_epochs=num_epochs,
			num_eval_steps_per_epoch=400,
			num_expl_steps_per_train_loop=path_length,
			num_trains_per_train_loop=100,
			# collect_new_paths=False,
		),

		load_demos=True,
		demo_paths=[
					# os.path.join(main_dir, "demos", f"{args.env_name}_model_noisy_9500_success.npy"),
					# os.path.join(main_dir, "demos", f"{args.env_name}_model_on_policy_1000_debug1.npy"),
					os.path.join(main_dir, "demos", f"Bottle_model_on_policy_1000_model1.npy"),
					# os.path.join(main_dir, "demos", f"{args.env_name}_model_on_policy_15000_all1.npy"),
					# os.path.join(main_dir, "demos", f"{args.env_name}_model_off_policy_4000_fail_1.npy"),
					],
		pretrain_rf=False,
		pretrain=True,

		env_kwargs={'config':dict(
			env_name='Bottle',
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03, frame_skip=5),
			# env_kwargs=dict(path_length=path_length, frame_skip=5),

			oracle='model',
			oracle_kwargs=dict(),
			input_in_obs=False,
			action_type='disc_traj',
			smooth_alpha=.8,

			adapts=['high_dim_user', 'reward'],
			apply_projection=False,
			space=0,
			num_obs=10,
			num_nonnoop=0,
			reward_max=0,
			reward_min=-1,
			input_penalty=1,
			reward_type='sparse',
		)},
	)
	search_space = {
		'seedid': [2000,2001],

		'env_kwargs.config.oracle_kwargs.threshold': [.5],
		'env_kwargs.config.state_type': [0],
		'env_kwargs.config.env_kwargs.debug': [False],

		'false_prop': [.5],
		'demo_path_proportions':[[1000]],
		# 'demo_path_proportions':[[15000]],
		'trainer_kwargs.policy_lr': [1e-3],
		'trainer_kwargs.use_mixup': [True,],
		'layer_norm': [True],
		
		# 'dropout_p': [.1,.05],
	}

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)
	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	def process_args(variant):
		variant['qf_lr'] = variant['trainer_kwargs']['policy_lr']
		variant['env_kwargs']['config']['seedid'] = variant['seedid']
		if not args.use_ray:
			variant['render'] = args.no_render
	args.process_args = process_args

	run_exp(experiment, variants, args)
