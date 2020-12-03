import rlkit.torch.pytorch_util as ptu
from rlkit.envs.make_env import make
from rlkit.torch.networks import Mlp

from copy import deepcopy

from utils import *
from policies import DemonstrationPolicy,BoltzmannPolicy,OverridePolicy,ComparisonMergePolicy
from full_path_collector import FullPathCollector

def experiment(variant):
	env = variant['env_class'](variant['env_kwargs'])
	env.seed(variant['seedid'])

	qf_kwargs = variant["qf_kwargs"]
	obs_dim = env.observation_space.low.size
	action_dim = env.action_space.low.size
	M = qf_kwargs['layer_size']
	qf1 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_sizes=[M,M,M],
		output_activation=Clamp(max=0),
	)
	qf2 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_sizes=[M,M,M],
		output_activation=Clamp(max=0),
	)
	target_qf1 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_sizes=[M,M,M],
		output_activation=Clamp(max=0),
	)
	target_qf2 = Mlp(
		input_size=obs_dim,
		output_size=action_dim,
		hidden_sizes=[M,M,M],
		output_activation=Clamp(max=0),
	)
	policy_kwargs = variant['policy_kwargs']
	eval_policy = BoltzmannPolicy(
		qf1,qf2,
		logit_scale=1e4
	)
	eval_path_collector = FullTrajPathCollector(
		env,
		eval_policy,
	)
	expl_policy = BoltzmannPolicy(
		qf1,qf2,
		logit_scale=variant['expl_kwargs']['logit_scale'])
	if variant['exploration_strategy'] == 'merge_arg':
		expl_policy = ComparisonMergePolicy(env.rng,expl_policy)
	elif variant['exploration_strategy'] == 'override':
		expl_policy = OverridePolicy(env,expl_policy)
	expl_path_collector = FullTrajPathCollector(
		env,
		expl_policy,
	)
	if variant.get('load_demos', False):
		path_loader_kwargs = variant.get("path_loader_kwargs", {})
		path_loader_class = variant.get('path_loader_class', AdaptPathLoader)
		path_loader = path_loader_class(trainer,
			replay_buffer=replay_buffer,
			demo_train_buffer=replay_buffer,
			demo_test_buffer=replay_buffer,
			**path_loader_kwargs
		)
		path_loader.load_demos()
	replay_buffer = AdaptReplayBuffer(
        variant['replay_buffer_size'],
        env,
    )
	trainer = DDQNTrainer(
		qf1=qf1,
		qf2=qf2,
		target_qf1=target_qf1,
		target_qf2=target_qf2,
		**variant['trainer_kwargs']
		)	
	algorithm = TorchBatchRLAlgorithm(
		trainer=trainer,
		exploration_env=env,
		evaluation_env=env,
		exploration_data_collector=expl_path_collector,
		evaluation_data_collector=eval_path_collector,
		replay_buffer=replay_buffer,
		**variant['algorithm_kwargs']
	)
	algorithm.to(ptu.device)
	if variant['pretrain']:
		for _ in range(variant['num_pretrain_loops']):
			train_data = replay_buffer.random_batch(variant['algorithm_kwargs']['batch_size'])
			trainer.train(train_data)
	if variant.get('render',False):
		env.render('human')
	algorithm.train()
	