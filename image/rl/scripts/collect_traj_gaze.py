from rl.policies import DemonstrationPolicy, UserInputPolicy, ArgmaxPolicy, FollowerPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.path_collectors import gaze_rollout
import rlkit.pythonplusplus as ppp

import os
from pathlib import Path
import argparse
import numpy as np
import rlkit.torch.pytorch_util as ptu
from copy import deepcopy
from gaze_capture.ITrackerModel import ITrackerModel
import torch
import math

import torch as th
from types import MethodType


def collect_demonstrations(variant):
	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid']+100)

	file_name = os.path.join(variant['eval_path'])
	qf1 = th.load(file_name,map_location=th.device("cpu"))['trainer/qf']

	policy = ArgmaxPolicy(
		qf=qf1,
	)
	# policy = FollowerPolicy(env)
	path_collector = FullPathCollector(
		env,
		DemonstrationPolicy(policy,env,p=variant['p']),
		rollout_fn=gaze_rollout
	)

	env.render('human')
	paths = []
	success_count = 0
	while len(paths) < variant['num_episodes']:
		target_index = 0
		# while target_index == 0:
		while target_index < env.base_env.num_targets:
			def set_target_index(self):
				self.target_index = target_index
			env.base_env.set_target_index = MethodType(set_target_index,env.base_env)
			collected_paths = path_collector.collect_new_paths(
				variant['path_length'],
				variant['path_length'],
			)
			# success_found = False
			for path in collected_paths:
				# if path['env_infos'][-1]['task_success'] == variant['on_policy']:
				paths.append(path)
				success_count += path['env_infos'][-1]['task_success']
				# for info in path['env_infos']:
				# 	info['gaze'] = False
			# if success_found:
			target_index += 1
			print("total paths collected: ", len(paths), "successes: ", success_count)
	return paths

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name',)
	parser.add_argument('--no_render', action='store_false')
	parser.add_argument('--use_ray', action='store_true')
	args, _ = parser.parse_known_args()
	main_dir = str(Path(__file__).resolve().parents[2])
	print(main_dir)

	path_length = 200
	variant = dict(
		seedid=3000,
		eval_path=os.path.join(main_dir,'util_models','cql_transfer.pkl'),
		env_kwargs={'config':dict(
			env_name='Bottle',
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03,frame_skip=5,stochastic=True),
			oracle='model',
			oracle_kwargs=dict(
				threshold=.5,
			),
			action_type='disc_traj',
			smooth_alpha=.8,

			# adapts = [],
			adapts = ['high_dim_user','reward'],
			gaze_dim=50,
			state_type=0,
			apply_projection=False,
			reward_max=0,
			reward_min=-1,
			input_penalty=1,
			reward_type='sparse',
		)},
		render = args.no_render and (not args.use_ray),

		on_policy=True,
		p=1,
		num_episodes=10,
		path_length=path_length,
		save_name_suffix="debug"
	)
	search_space = {
		'env_kwargs.config.oracle_kwargs.epsilon': 0 if variant['on_policy'] else .7, # higher epsilon = more noise
	}
	search_space = ppp.dot_map_dict_to_nested_dict(search_space)
	variant = ppp.merge_recursive_dicts(variant,search_space)

	def process_args(variant):
		variant['env_kwargs']['config']['seedid'] = variant['seedid']
		variant['save_name'] = f"{variant['env_kwargs']['config']['env_name']}_{variant['env_kwargs']['config']['oracle']}"\
								+ f"_{'on_policy' if variant['on_policy'] else 'off_policy'}_{variant['num_episodes']}"\
								+ "_" + variant['save_name_suffix']

	
	import time
	current_time = time.time_ns()
	variant['seedid'] = current_time
	process_args(variant)
	paths = collect_demonstrations(variant)

	i_tracker = ITrackerModel()
	if torch.cuda.is_available():
		device = torch.device("cuda:0")
		i_tracker.cuda()
		state = torch.load(os.path.join(main_dir,'gaze_capture','checkpoint.pth.tar'))['state_dict']
	else:
		device = "cpu"
		state = torch.load(os.path.join(main_dir,'gaze_capture','checkpoint.pth.tar'),
						map_location=torch.device('cpu'))['state_dict']
	i_tracker.load_state_dict(state, strict=False)

	for path in paths:
		data = []
		gazes = [info['gaze_features'] for info in path['env_infos']]
		if len(gazes) > 0:
			point = zip(*gazes)
			point = [torch.from_numpy(np.array(feature)).float().to(device) for feature in point]

			batch_size = 32
			n_batches = math.ceil(len(point[0]) / batch_size)
			for j in range(n_batches):
				batch = [feature[j * batch_size: (j + 1) * batch_size] for feature in point]
				output = i_tracker(*batch)
				data.extend(output.detach().cpu().numpy())

		for info,point in zip(path['env_infos'],data):
			info['gaze_features'] = point

	np.save(os.path.join(main_dir,"demos",variant['save_name']+f"_{variant['seedid']}"), paths)
