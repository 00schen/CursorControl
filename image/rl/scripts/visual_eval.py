from rl.policies import BoltzmannPolicy
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

	file_name = os.path.join(variant['eval_path'])
	qf1 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf1']

	policy = BoltzmannPolicy(
		qf1=qf1,
		logit_scale=1e4,
	)
	# policy = th.load(file_name,map_location=th.device("cpu"))['trainer/policy']
	eval_path_collector = FullPathCollector(
		env,
		policy,
	)

	if variant.get('render',False):
		env.render('human')
	eval_collected_paths = eval_path_collector.collect_new_paths(
		variant['path_length'],
		variant['path_length']*variant['num_evals'],
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
		eval_path=os.path.join(main_dir,'logs','test-b-ground-truth-offline-3','test_b_ground_truth_offline_3_2021_01_22_16_10_53_0000--s-0','params.pkl'),
		env_kwargs={'config':dict(
			env_name='Bottle',
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03,frame_skip=5,capture_frames=False),
			# env_kwargs=dict(path_length=path_length,frame_skip=5),

			oracle='model',
			oracle_kwargs=dict(threshold=.5),
			action_type='disc_traj',
			smooth_alpha = .8,

			adapts = ['high_dim_user','reward'],
			state_type=2,
			apply_projection=False,
			reward_max=0,
			reward_min=-1,
			input_penalty=1,
			reward_type='sparse',
		)},
		render = args.no_render,
		num_evals = 100,
	)

	def process_args(variant):
		variant['env_kwargs']['config']['seedid'] = variant['seedid']

	process_args(variant)
	evaluation(variant)