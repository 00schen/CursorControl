from rl.policies import EncDecPolicy, DemonstrationPolicy, KeyboardPolicy
from rl.oracles import BottleOracle, KitchenOracle
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
import rlkit.pythonplusplus as ppp

import os
from pathlib import Path
import argparse
import numpy as np
from copy import deepcopy

import torch as th
from types import MethodType


def collect_demonstrations(variant):
    import time

    current_time = time.time_ns()
    env = default_overhead(variant['env_kwargs']['config'])
    env.seed(variant['seedid'] + current_time)

    if variant['oracle'] == 'model':
        file_name = os.path.join(variant['eval_path'])
        loaded = th.load(file_name, map_location='cpu')
        policy = EncDecPolicy(
            policy=loaded['trainer/policy'],
            features_keys=list(env.feature_sizes.keys()),
            vae=loaded['trainer/vae'],
            incl_state=False,
            sample=False,
            deterministic=False
        )
    elif variant['oracle'] == 'scripted':
        policy = BottleOracle()
        policy = DemonstrationPolicy(policy, env, p=variant['p'])
    elif variant['oracle'] == 'keyboard':
        policy = KeyboardPolicy()
        policy = DemonstrationPolicy(policy, env, p=variant['p'])

    # print(loaded['trainer/policy'])
    # print(loaded['trainer/vae'].encoder)

    path_collector = FullPathCollector(
        env,
        policy
    )

    if variant.get('render', False):
        env.render('human')
    paths = []
    success_count = 0
    while len(paths) < variant['num_episodes']:
        target_index = 0
        while target_index < env.base_env.num_targets:
            def set_target_index(self):
                self.target_index = target_index

            env.base_env.set_target_index = MethodType(set_target_index, env.base_env)
            collected_paths = path_collector.collect_new_paths(
                variant['path_length'],
                variant['path_length'],
            )
            success_found = False
            for path in collected_paths:
                if path['env_infos'][-1]['task_success']:
                # if sum(path['env_infos'][-1]['tasks']) > 2:
                    paths.append(path)
                    success_count += path['env_infos'][-1]['task_success']
                    success_found = True
            if success_found:
                target_index += 1
            print("total paths collected: ", len(paths), "successes: ", success_count)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', )
    parser.add_argument('--suffix', default='test')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    args, _ = parser.parse_known_args()
    main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200
    variant = dict(
        seedid=3000,
        eval_path=os.path.join(main_dir, 'util_models', 'kitchen-debug.pkl'),
        env_kwargs={'config': dict(
            env_name='Bottle',
            step_limit=path_length,
            # env_kwargs=dict(pretrain_assistance=True, debug=True, target_indices=[0]),
            env_kwargs=dict(),
            action_type='disc_traj',
            smooth_alpha=1,

            factories=[],
            adapts=['goal',],
            terminate_on_failure=False,
            goal_noise_std=0,
        )},
        render=args.no_render and (not args.use_ray),

        oracle='scripted',

        p=1,
        num_episodes=5000,
        path_length=path_length,
        save_name_suffix=args.suffix,

    )
    # search_space = {
    # }
    # search_space = ppp.dot_map_dict_to_nested_dict(search_space)
    # variant = ppp.merge_recursive_dicts(variant, search_space)


    def process_args(variant):
        variant['env_kwargs']['config']['seedid'] = variant['seedid']
        variant['save_name'] = f"{variant['env_kwargs']['config']['env_name']}" \
                           + f"_{variant['oracle']}" \
                           + f"_{variant['num_episodes']}" \
                           + "_" + variant['save_name_suffix']


    if args.use_ray:
        import ray
        from itertools import count

        ray.init(_temp_dir='/tmp/ray_exp1', num_gpus=0)

        process_args(variant)

        @ray.remote(num_cpus=1, num_gpus=0)
        class Sampler:
            def sample(self, variant):
                variant = deepcopy(variant)
                return collect_demonstrations(variant)

        num_workers = 10
        variant['num_episodes'] = variant['num_episodes'] // num_workers

        samplers = [Sampler.remote() for i in range(num_workers)]
        samples = [samplers[i].sample.remote(variant) for i in range(num_workers)]
        samples = [ray.get(sample) for sample in samples]
        paths = list(sum(samples, []))
        np.save(os.path.join(main_dir, "demos", variant['save_name']), paths)
    else:
        import time

        current_time = time.time_ns()
        variant['seedid'] = current_time
        process_args(variant)
        paths = collect_demonstrations(variant)
        np.save(os.path.join(main_dir, "demos", variant['save_name']), paths)
