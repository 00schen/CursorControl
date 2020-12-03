# from rl.policies import BoltzmannPolicy,OverridePolicy,ComparisonMergePolicy
from rl.path_collectors import FullPathCollector
from rl.env_wrapper import default_overhead
from rl.simple_path_loader import SimplePathLoader

import os
from pathlib import Path
import argparse
import numpy as np
import torch.optim as optim

import torch as th

def evaluation(variant):
	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid']+10)

	file_name = os.path.join(variant['eval_path'],'params.pkl')
	# qf1 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf1']
	# qf2 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf2']

	# policy_kwargs = variant['policy_kwargs']
	# policy = BoltzmannPolicy(
	# 	qf1=qf1,
	# 	qf2=qf2,
	# 	**policy_kwargs,
	# )
	policy = th.load(file_name,map_location=th.device("cpu"))['trainer/policy']
	eval_path_collector = FullPathCollector(
		env,
		policy,
	)

	if variant.get('render',False):
		env.render('human')
	eval_collected_paths = eval_path_collector.collect_new_paths(
		variant['path_length'],
		variant['path_length']*10,
		discard_incomplete_paths=False,
	)

	np.save(os.path.join(variant['save_path'],'evaluated_paths'), eval_collected_paths)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name',)
	parser.add_argument('--exp_name', default='a-test')
	parser.add_argument('--no_render', action='store_false')
	parser.add_argument('--use_ray', action='store_true')
	parser.add_argument('--gpus', default=0, type=int)
	parser.add_argument('--per_gpu', default=1, type=int)
	args, _ = parser.parse_known_args()
	main_dir = str(Path(__file__).resolve().parents[2])
	print(main_dir)

	path_length = 200
	variant = dict(
		seedid=2002,
		path_length=path_length,
		eval_path=os.path.join(main_dir,'logs','testli_sac1_2020_12_03_01_22_32_0000--s-0'),
		env_kwargs={'config':dict(
			env_name=args.env_name,
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03,frame_skip=5),
			# env_kwargs=dict(path_length=path_length,frame_skip=5),
			smooth_alpha=.8,

			oracle='model',
			oracle_kwargs=dict(),
			action_type='trajectory',

			adapts = ['stack','reward'],
			space=0,
			num_obs=10,
			num_nonnoop=10,
			reward_max=0,
			reward_min=-np.inf,
			input_penalty=1,
		)},
		render = args.no_render,
	)

	def process_args(variant):
		variant['env_kwargs']['config']['seedid'] = variant['seedid']

	process_args(variant)
	evaluation(variant)