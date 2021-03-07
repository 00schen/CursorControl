import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp, ConcatMlp
from rlkit.torch.networks import Clamp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import DummyPolicy
from rl.path_collectors import FullPathCollector, CustomPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.trainers import RewardTrainer
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
	rf = ConcatMlp(
		input_size=obs_dim *2+action_dim,
		output_size=1,
		hidden_sizes=[M, M, M, M],
		# hidden_activation=make_alpha_relu(variant['dropout_p']),
		hidden_activation=F.leaky_relu,
		layer_norm=True,
		output_activation=Clamp(max=5,min=-5),
	)
	eval_policy = DummyPolicy(
		env,
	)
	eval_path_collector = FullPathCollector(
		env,
		eval_policy,
		save_env_in_snapshot=False
	)
	trainer = RewardTrainer(
		rf=rf,
		**variant['trainer_kwargs']
	)
	replay_buffer = BalancedReplayBuffer(
		variant['replay_buffer_size'],
		env,
		false_prop=variant['false_prop'],
		target_name=variant['trainer_kwargs']['target_name'],
		env_info_sizes={'noop':1,'episode_success':1}
	)
	algorithm = TorchBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=env,
		evaluation_env=env,
		exploration_data_collector=eval_path_collector,
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
	num_epochs = int(1e3)
	variant = dict(
		layer_size=256,
		replay_buffer_size=int(2e4)*path_length,
		# intervention_prop=.5
		trainer_kwargs=dict(
		),
		algorithm_args=dict(
			batch_size=256,
			max_path_length=1,
			eval_path_length=1,
			num_epochs=num_epochs,
			num_eval_steps_per_epoch=1,
			num_expl_steps_per_train_loop=1,
			num_trains_per_train_loop=1,
			collect_new_paths=False,
			eval_paths=False,
		),

		load_demos=True,
		demo_paths=[
					# os.path.join(main_dir, "demos", f"{args.env_name}_model_noisy_9500_success.npy"),
					# os.path.join(main_dir, "demos", f"{args.env_name}_model_off_policy_10000_p_.7_eps_.5_1.npy"),
					os.path.join(main_dir, "demos", f"Bottle_model_on_policy_15000_model1.npy"),
					# os.path.join(main_dir, "demos", f"{args.env_name}_model_on_policy_15000_all1.npy"),
					# os.path.join(main_dir, "demos", f"{args.env_name}_dummy_1000.npy"),
					# os.path.join(main_dir, "demos", f"{args.env_name}_model_off_policy_4000_fail_1.npy"),
					],

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
		'seedid': [2000, 2001, 2002, 2004, 2005],

		'env_kwargs.config.oracle_kwargs.threshold': [.5],
		'env_kwargs.config.state_type': [0],

		'false_prop': [.5,],
		'demo_path_proportions':[[int(1.5e4)], ],
		'trainer_kwargs.target_name': ['noop'],
		'trainer_kwargs.rf_lr': [1e-4],
		'trainer_kwargs.use_mixup': [True],
		'dropout_p': [.05],
	}


	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)
	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	def process_args(variant):
		variant['qf_lr'] = variant['trainer_kwargs']['rf_lr']
		variant['env_kwargs']['config']['seedid'] = variant['seedid']
		if not args.use_ray:
			variant['render'] = args.no_render
	args.process_args = process_args

	run_exp(experiment, variants, args)
