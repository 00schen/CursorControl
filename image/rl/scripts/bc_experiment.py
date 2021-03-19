import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import VAEGazePolicy, MlpPolicy, VAEMixedPolicy
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.balanced_replay_buffer import BalancedReplayBuffer

from rl.env_wrapper import default_overhead
from rl.simple_path_loader import SimplePathLoader
from rl.trainers import DiscreteVAEBCTrainerTorch, DiscreteBCTrainerTorch, DiscreteMixedVAEBCTrainerTorch
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rl.path_collectors import FullPathCollector
from rl.policies import ArgmaxPolicy

import os
import torch
import gtimer as gt
from pathlib import Path
from rlkit.launchers.launcher_util import setup_logger, reset_execution_environment
import argparse


def experiment(variant, logdir):
    from rlkit.core import logger
    gaze_dim = 128
    embedding_dim = 3
    env = default_overhead(variant['env_kwargs']['config'])
    env.seed(variant['seedid'])
    M = variant["layer_size"]
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    policy_type = VAEMixedPolicy
    policy = policy_type(input_size=obs_dim,
                         output_size=action_dim,
                         decoder_hidden_sizes=[M, M, M, M],
                         layer_norm=True,
                         gaze_dim=gaze_dim,
                         embedding_dim=embedding_dim,
                         num_encoders=1,
                         encoder_hidden_sizes=(64,)
                         )
    policy.to(ptu.device)

    trainer_type = DiscreteMixedVAEBCTrainerTorch
    trainer = trainer_type(
        policy=policy,
        policy_lr=variant['trainer_kwargs']['lr']
    )

    gaze_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        env,
    )

    prior_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        env,
    )

    replay_buffer = BalancedReplayBuffer(
        gaze_buffer, prior_buffer
    )

    path_loader = SimplePathLoader(
        demo_path=variant['demo_paths'],
        demo_path_proportion=variant['demo_path_proportions'],
        replay_buffers=[gaze_buffer, prior_buffer],
    )
    path_loader.load_demos()

    eval_policy = ArgmaxPolicy(
        policy
    )

    eval_path_collector = FullPathCollector(
        env,
        eval_policy,
        save_env_in_snapshot=False
    )

    expl_policy = eval_policy

    expl_path_collector = FullPathCollector(
        env,
        expl_policy,
        save_env_in_snapshot=False
    )

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
    torch.save(policy, pretrain_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', )
    parser.add_argument('--exp_name', default='bc_new_gaze')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    args, _ = parser.parse_known_args()
    main_dir = str(Path(__file__).resolve().parents[2])
    print(main_dir)

    path_length = 200
    num_epochs = int(1e4)
    variant = dict(
        layer_size=128,
        replay_buffer_size=int(1e6),
        trainer_kwargs=dict(
            lr=1e-3,
            aux_loss_weight=1,
            discrim_hidden=(64,),
            l2_weight=0,
            reconstruct_weight=1
        ),
        bc_args=dict(
            batch_size=128,
            max_path_length=path_length,
            num_epochs=100,
            num_eval_steps_per_epoch=path_length * 3,
            num_expl_steps_per_train_loop=0,
            collect_new_paths=False,
            num_trains_per_train_loop=1000,
        ),
        demo_paths=[
            os.path.join(main_dir, "demos",
                         f"int_OneSwitch_sim_gaze_on_policy_100_all_debug_1616176591751061503.npy"),
            os.path.join(main_dir, "demos",
                         f"int_OneSwitch_sim_goal_model_on_policy_1000_all_debug_1615835470059229510.npy"),
        ],


        env_kwargs={'config': dict(
            env_name=args.env_name,
            step_limit=path_length,
            env_kwargs=dict(success_dist=.03, frame_skip=5, stochastic=True),
            oracle='sim_gaze_model',
            oracle_kwargs=dict(),
            gaze_oracle_kwargs={'mode': 'eval'},
            action_type='disc_traj',
            smooth_alpha=.8,

            adapts=['high_dim_user', 'reward'],
            space=0,
            num_obs=10,
            num_nonnoop=0,
            reward_max=0,
            reward_min=-1,
            reward_type='user_penalty',
            input_in_obs=True,
            input_penalty=1,
            apply_projection=False,
        )},

        seedid=2000,
        demo_path_proportions=[100, 1000],

    )

    variants = [variant]


    def process_args(variant):
        variant['env_kwargs']['config']['seedid'] = variant['seedid']


    if args.use_ray:
        import ray
        from ray.util import ActorPool
        from itertools import count

        ray.init(num_gpus=args.gpus)


        @ray.remote
        class Iterators:
            def __init__(self):
                self.run_id_counter = count(0)

            def next(self):
                return next(self.run_id_counter)


        iterator = Iterators.options(name="global_iterator").remote()


        @ray.remote(num_cpus=1, num_gpus=1 / args.per_gpu if args.gpus else 0)
        class Runner:
            def run(self, variant):
                gt.reset_root()
                ptu.set_gpu_mode(True)
                process_args(variant)
                iterator = ray.get_actor("global_iterator")
                run_id = ray.get(iterator.next.remote())
                save_path = os.path.join(main_dir, 'logs')
                reset_execution_environment()
                log_dir = setup_logger(exp_prefix=args.exp_name, variant=variant, base_log_dir=save_path, exp_id=run_id, )
                experiment(variant, log_dir)


        runners = [Runner.remote() for i in range(args.gpus * args.per_gpu)]
        runner_pool = ActorPool(runners)
        list(runner_pool.map(lambda a, v: a.run.remote(v), variants))
    else:
        import time

        current_time = time.time_ns()
        variant = variants[0]
        run_id = 0
        save_path = os.path.join(main_dir, 'logs')
        logdir = setup_logger(exp_prefix=args.exp_name, variant=variant, base_log_dir=save_path, exp_id=run_id)
        process_args(variant)
        experiment(variant, logdir)

