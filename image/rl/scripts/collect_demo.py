from rl.policies import DemonstrationPolicy
from rl.path_collectors import FullPathCollector
from rl.env_wrapper import default_overhead
from rl.simple_path_loader import SimplePathLoader
import rlkit.pythonplusplus as ppp

import os
from pathlib import Path
import argparse
import numpy as np
import torch.optim as optim
from copy import deepcopy

import torch as th
from types import MethodType
def collect_demonstrations(variant):
	env = default_overhead(variant['env_kwargs']['config'])
	env.seed(variant['seedid']+100)

	path_collector = FullPathCollector(
		env,
		DemonstrationPolicy(env,p=variant['p_on'] if variant['on_policy'] else variant['p_off']),
	)

	if variant.get('render',False):
		env.render('human')
	paths = []
	success_count = 0
	while len(paths) < variant['num_episodes']:
		target_index = 0
		while target_index < env.base_env.num_targets:
			def set_target_index(self):
				self.target_index = target_index
			env.base_env.set_target_index = MethodType(set_target_index,env.base_env)
			collected_paths = path_collector.collect_new_paths(
				variant['path_length'],
				variant['path_length'],
			)
			success_found = False
			for path in collected_paths:
				# if path['env_infos'][-1]['task_success'] == variant['on_policy']:
				if True:
					paths.append(path)
					success_found = True
			if success_found:
				target_index += 1
			print("total paths collected: ", len(paths))
	return paths

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name',)
	parser.add_argument('--exp_name', default='a-test')
	parser.add_argument('--no_render', action='store_false')
	parser.add_argument('--use_ray', action='store_true')
	args, _ = parser.parse_known_args()
	main_dir = str(Path(__file__).resolve().parents[2])
	print(main_dir)

	path_length = 100
	variant = dict(
		env_kwargs={'config':dict(
			env_name=args.env_name,
			step_limit=path_length,
			env_kwargs=dict(success_dist=.03,frame_skip=5),

			oracle='model',
			oracle_kwargs=dict(),
			input_in_obs=True,
			action_type='disc_traj',

			adapts = [],
		)},
		render = args.no_render and (not args.use_ray),

		on_policy=False,
		p_on=.99,
		p_off=.5,
		num_episodes=5000,
		path_length=path_length,
		save_name_suffix="all_8"
	)
	search_space = {
		'seedid': 1000,
		'env_kwargs.config.smooth_alpha': .8,
		'env_kwargs.config.oracle_kwargs.threshold': .5,
		'env_kwargs.config.oracle_kwargs.epsilon': 0,
	}
	search_space = ppp.dot_map_dict_to_nested_dict(search_space)
	variant = ppp.merge_recursive_dicts(variant,search_space)

	def process_args(variant):
		variant['env_kwargs']['config']['seedid'] = variant['seedid']
		if variant['render']:
			variant['num_episodes'] = 20
		variant['save_name'] = f"{args.env_name}_{variant['env_kwargs']['config']['oracle']}"\
								+ f"_{'on_policy' if variant['on_policy'] else 'off_policy'}_{variant['num_episodes']}"\
								+ "_" + variant['save_name_suffix']

	if args.use_ray:
		import ray
		from ray.util import ActorPool
		from itertools import cycle,count
		ray.init(temp_dir='/tmp/ray_exp', num_gpus=0)

		@ray.remote
		class Iterators:
			def __init__(self):
				self.run_id_counter = count(0)
			def next(self):
				return next(self.run_id_counter)
		iterator = Iterators.options(name="global_iterator").remote()

		process_args(variant)
		@ray.remote(num_cpus=1,num_gpus=0)
		class Sampler:
			def sample(self,variant):
				variant = deepcopy(variant)
				variant['seedid'] += ray.get(iterator.next.remote())
				return collect_demonstrations(variant)
		num_workers = 10
		variant['num_episodes'] = variant['num_episodes']//num_workers

		samplers = [Sampler.remote() for i in range(num_workers)]
		samples = [samplers[i].sample.remote(variant) for i in range(num_workers)]
		samples = [ray.get(sample) for sample in samples]
		paths = list(sum(samples,[]))
		np.save(os.path.join(main_dir,"demos",variant['save_name']), paths)
	else:
		import time
		current_time = time.time_ns()
		variant['seedid'] = current_time
		process_args(variant)
		paths = collect_demonstrations(variant)
		np.save(os.path.join(main_dir,"demos",variant['save_name']+f"_{variant['seedid']}"), paths)
