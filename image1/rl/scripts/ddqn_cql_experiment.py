import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp, ConcatMlp
from rlkit.torch.networks import Clamp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import BoltzmannPolicy, OverridePolicy, ComparisonMergePolicy, ArgmaxPolicy
from rl.path_collectors import FullPathCollector, CustomPathCollector, rollout
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.trainers import DDQNCQLTrainer
from rl.misc.balanced_replay_buffer import BalancedReplayBuffer
from rl.scripts.run_util import run_exp
from rl.misc.reward_ensemble import RewardEnsemble

import os
from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
from torch.nn import functional as F


def experiment(variant):
	from rlkit.core import logger
	import torch as th

	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid'])

	obs_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size
	M = variant["layer_size"]
	# upper_q = variant['env_kwargs']['config']['reward_max']*variant['env_kwargs']['config']['step_limit']
	# lower_q = variant['env_kwargs']['config']['reward_min']*variant['env_kwargs']['config']['step_limit']
	upper_q = 0
	lower_q = -500
	qf = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_sizes=[M, M, M, M],
		hidden_activation=F.leaky_relu,
		layer_norm=True,
		# output_activation=Clamp(max=0, min=-5e3),
		output_activation=Clamp(max=upper_q, min=lower_q),
	)
	target_qf = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_sizes=[M, M, M, M],
		hidden_activation=F.leaky_relu,
		layer_norm=True,
		# output_activation=Clamp(max=0, min=-5e3),
		output_activation=Clamp(max=upper_q, min=lower_q),
	)
	rf = RewardEnsemble(variant['rf_path'])
	# rf = th.load(variant['rf_path'],map_location=ptu.device)['trainer/rf']
	eval_policy = ArgmaxPolicy(
		qf,
	)
	# eval_path_collector = CustomPathCollector(
	eval_path_collector = FullPathCollector(
		env,
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
		# rollout_fn=rollout,
	)
	trainer = DDQNCQLTrainer(
		qf=qf,
		target_qf=target_qf,
		rf=rf,
		**variant['trainer_kwargs']
		)
	replay_buffer = BalancedReplayBuffer(
		variant['replay_buffer_size'],
		env,
		target_name='task_success',
		false_prop=variant['false_prop'],
		env_info_sizes={'task_success':1},
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
	if variant['pretrain_rf']:
		logger.remove_tabular_output(
			'progress.csv', relative_to_snapshot_dir=True,
		)
		logger.add_tabular_output(
			'pretrain_rf.csv', relative_to_snapshot_dir=True,
		)
		from tqdm import tqdm
		for _ in tqdm(range(int(1e5)), miniters=10, mininterval=10):
			train_data = replay_buffer.random_batch(variant['algorithm_args']['batch_size'])
			trainer.pretrain_rf(train_data)
		logger.remove_tabular_output(
			'pretrain_rf.csv', relative_to_snapshot_dir=True,
		)
		logger.add_tabular_output(
			'progress.csv', relative_to_snapshot_dir=True,
		)

	if variant['pretrain']:
		import gtimer as gt
		import torch as th
		logger.remove_tabular_output(
			'progress.csv', relative_to_snapshot_dir=True,
		)
		logger.add_tabular_output(
			'pretrain.csv', relative_to_snapshot_dir=True,
		)
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
		logger.remove_tabular_output(
			'pretrain.csv', relative_to_snapshot_dir=True,
		)
		logger.add_tabular_output(
			'progress.csv', relative_to_snapshot_dir=True,
		)
		pretrain_file_path = os.path.join(logger.get_snapshot_dir(), 'pretrain.pkl')
		th.save(trainer.get_snapshot(), pretrain_file_path)
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
	num_epochs = int(10)
	variant = dict(
		layer_size=128,
		rf_path=os.path.join(main_dir, 'logs', 'test-b-reward-1'),
		exploration_argmax=True,
		exploration_strategy='',
		expl_kwargs=dict(
			logit_scale=1000,
		),
		replay_buffer_size=int(2e4)*path_length,
		# intervention_prop=.5
		trainer_kwargs=dict(
			# qf_lr=1e-3,
			soft_target_tau=1e-2,
			target_update_period=1,
			# reward_update_period=int(1e8),
			qf_criterion=None,

			discount=1-(1/path_length),
			reward_scale=1.0,

			# temp=1.0,
			# min_q_weight=1.0,
			# ground_truth=False,
			target_name='noop',
			add_ood_term=-1,
		),
		# algorithm_args=dict(
		# 	batch_size=256,
		# 	max_path_length=path_length,
		# 	eval_path_length=1,
		# 	num_epochs=0,
		# 	num_eval_steps_per_epoch=1,
		# 	num_expl_steps_per_train_loop=path_length,
		# 	num_trains_per_train_loop=5,
		# ),
		algorithm_args=dict(
			batch_size=256,
			max_path_length=path_length,
			num_epochs=int(3e4),
			num_eval_steps_per_epoch=path_length,
			num_expl_steps_per_train_loop=0,
			collect_new_paths=False,
			num_trains_per_train_loop=1000,
		),

		load_demos=True,
		demo_paths=[
					os.path.join(main_dir, "demos", f"Bottle_model_on_policy_15000_model1.npy"),
					os.path.join(main_dir, "demos", f"Bottle_model_noisy_9500_success.npy"),
					# os.path.join(main_dir, "demos", f"OneSwitch_model_on_policy_10000_model.npy"),
					],
		pretrain_rf=False,
		pretrain=False,

		env_kwargs={'config':dict(
			# env_name=args.env_name,
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03, frame_skip=5,),

			oracle='model',
			oracle_kwargs=dict(
				threshold=.5
			),
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
			reward_type='part_sparse',
		)},

		logprob=False,
	)
	search_space = {
		'seedid': [2000,2001],

		'trainer_kwargs.ground_truth': [True,],
		'trainer_kwargs.temp': [1],
		'trainer_kwargs.min_q_weight': [1],
		'trainer_kwargs.reward_bias': [.5],
		'env_kwargs.config.state_type': [1],
		'env_kwargs.config.env_kwargs.debug': [False],
		'env_kwargs.config.env_name': ['Bottle'],
		# 'env_kwargs.config.env_name': ['OneSwitch'],

		'demo_path_proportions':[[int(5.5e3),int(1e4)], ],
		# 'demo_path_proportions':[[int(1.5e4)], ],
		# 'demo_path_proportions':[[10], ],
		'false_prop': [.9],
		'trainer_kwargs.qf_lr': [1e-5],
	}

	sweeper = hyp.DeterministicHyperparameterSweeper(
		search_space, default_parameters=variant,
	)
	variants = []
	for variant in sweeper.iterate_hyperparameters():
		variants.append(variant)

	def process_args(variant):
		# variant['demo_paths'] = {
		# 	'Bottle': [variant['demo_paths'][0]],
		# 	'OneSwitch': [variant['demo_paths'][1]],
		# }[variant['env_kwargs']['config']['env_name']]
		variant['trainer_kwargs']['learning_rate'] = variant['trainer_kwargs'].pop('qf_lr')
		variant['qf_lr'] = variant['trainer_kwargs']['learning_rate']
		variant['env_kwargs']['config']['seedid'] = variant['seedid']
		if not args.use_ray:
			variant['render'] = args.no_render
	args.process_args = process_args

	run_exp(experiment, variants, args)
