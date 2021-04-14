import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlp, QrGazeMlp, QrMlp, MlpPolicy, MlpGazePolicy
from rlkit.torch.networks import Clamp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.balanced_replay_buffer import BalancedReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import BoltzmannPolicy, OverridePolicy, ComparisonMergePolicy, ArgmaxPolicy, UserInputPolicy
from rl.path_collectors import FullPathCollector
from rl.env_wrapper import default_overhead
from rl.simple_path_loader import SimplePathLoader
from rl.trainers import DDQNCQLTrainer, QRDDQNCQLTrainer

import os
import gtimer as gt
from pathlib import Path
from rlkit.launchers.launcher_util import setup_logger, reset_execution_environment
import rlkit.util.hyperparameter as hyp
import argparse
import torch as th
from torch.nn import functional as F
import copy


def experiment(variant):
    env = default_overhead(variant['env_kwargs']['config'])
    env.seed(variant['seedid'])

    eval_env_config = copy.deepcopy(variant['env_kwargs']['config'])
    eval_env_config['seedid'] += 100
    eval_env_config['gaze_oracle_kwargs']['mode'] = 'eval'

    eval_env = default_overhead(eval_env_config)
    eval_env.seed(variant['seedid'] + 100)

    gaze_vae = None
    if variant['gaze_vae_path'] is not None:
        gaze_vae = th.load(variant['gaze_vae_path'], map_location=th.device("cpu"))

    decoder = None
    if variant['decoder_path'] is not None:
        decoder = th.load(variant['decoder_path'], map_location=th.device("cpu"))['trainer/qf'].decoder

    if not variant['from_pretrain']:
        obs_dim = env.observation_space.low.size
        action_dim = env.action_space.low.size
        gaze_dim = 128
        embedding_dim = 3
        M = variant["layer_size"]
        qf = MlpGazePolicy(
            input_size=obs_dim,
            output_size=action_dim,
            encoder_hidden_sizes=[M],
            decoder_hidden_sizes=[M, M, M, M],
            hidden_activation=F.relu,
            layer_norm=True,
            gaze_vae=gaze_vae,
            gaze_dim=gaze_dim,
            embedding_dim=embedding_dim,
            decoder=decoder
        )
        target_qf = MlpGazePolicy(
            input_size=obs_dim,
            output_size=action_dim,
            encoder_hidden_sizes=[M],
            decoder_hidden_sizes=[M, M, M, M],
            hidden_activation=F.relu,
            layer_norm=True,
            gaze_vae=copy.deepcopy(gaze_vae),
            gaze_dim=gaze_dim,
            embedding_dim=embedding_dim,
            decoder=copy.deepcopy(decoder)
        )
        # qf = MlpPolicy(
        #     input_size=obs_dim,
        #     output_size=action_dim,
        #     hidden_sizes=[M, M, M, M],
        #     hidden_activation=F.relu,
        #     layer_norm=True,
        # )
        # target_qf = MlpPolicy(
        #     input_size=obs_dim,
        #     output_size=action_dim,
        #     hidden_sizes=[M, M, M, M],
        #     hidden_activation=F.relu,
        #     layer_norm=True,
        # )
        rf = ConcatMlp(
            input_size=obs_dim * 2,
            output_size=1,
            hidden_sizes=[M, M, M, M],
            hidden_activation=F.leaky_relu,
            layer_norm=True,
            output_activation=Clamp(max=-1e-2, min=-5),
        )
    else:
        pretrain_file_path = variant['pretrain_file_path']
        qf = th.load(pretrain_file_path, map_location=th.device("cpu"))['trainer/qf']
        target_qf = th.load(pretrain_file_path, map_location=th.device("cpu"))['trainer/target_qf']
        rf = th.load(pretrain_file_path, map_location=th.device("cpu"))['trainer/rf']

    if variant['freeze_encoder']:
        for param in qf.gaze_vae.parameters():
            param.requires_grad = False

    eval_policy = ArgmaxPolicy(
        qf
    )
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
            qf, eps=variant['expl_kwargs']['eps']
        )
    if variant['exploration_strategy'] == 'merge_arg':
        expl_policy = ComparisonMergePolicy(env.rng, expl_policy, env.oracle.size)
    elif variant['exploration_strategy'] == 'override':
        expl_policy = UserInputPolicy(env, p=1, base_policy=expl_policy, intervene=True)
    expl_path_collector = FullPathCollector(
        env,
        expl_policy,
        save_env_in_snapshot=False
    )
    trainer = DDQNCQLTrainer(
        qf=qf,
        target_qf=target_qf,
        rf=rf,
        **variant['trainer_kwargs']
    )

    online_buffer = EnvReplayBuffer(
         variant['replay_buffer_size'],
         env,
    )
    offline_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        env,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=online_buffer,
        **variant['algorithm_args']
    )
    algorithm.to(ptu.device)

    if variant.get('load_demos', False):
        path_loader = SimplePathLoader(
            demo_path=variant['demo_paths'],
            demo_path_proportion=variant['demo_path_proportions'],
            replay_buffers=[offline_buffer]
            # replay_buffers=[gaze_buffer, prior_buffer]
        )
        path_loader.load_demos()
    from rlkit.core import logger

    if variant['pretrain']:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'pretrain.csv', relative_to_snapshot_dir=True, )

        bc_algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=offline_buffer,
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

    trainer.min_q_weight = 0
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', )
    parser.add_argument('--exp_name', default='gaze_sanity')
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
        from_pretrain=False,
        pretrain_file_path=os.path.join(main_dir, 'logs', 'gaze-sanity', 'gaze_sanity_2021_04_09_21_31_21_0000--s-0',
                                        'params.pkl'),
        decoder_path=os.path.join(main_dir, 'logs', 'gaze-sanity', 'gaze_sanity_2021_04_10_10_02_49_0000--s-0',
                                        'params.pkl'),
        gaze_vae_path=os.path.join(main_dir, 'logs', 'gaze_vae', '2021-04-08--18-38-06.pkl'),
        layer_size=128,
        exploration_argmax=True,
        exploration_strategy='',
        expl_kwargs=dict(
            logit_scale=10,
        ),
        replay_buffer_size=int(5e4) * path_length,
        trainer_kwargs=dict(
            # qf_lr=1e-3,
            soft_target_tau=1e-2,
            target_update_period=1,
            # reward_update_period=int(1e8),
            qf_criterion=None,
            discount=0.99,
            reward_scale=1.0,
            # temp=1.0,
            # min_q_weight=1.0,
            beta=0.1,
            rew_class_weight=1,
            sample=False
        ),
        algorithm_args=dict(
            batch_size=128,
            max_path_length=path_length,
            eval_path_length=path_length,
            num_epochs=200,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=500,
            min_num_steps_before_training=1000,
            num_train_loops_per_epoch=10,
        ),
        bc_args=dict(
            batch_size=256,
            max_path_length=path_length,
            num_epochs=200,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=0,
            collect_new_paths=False,
            num_trains_per_train_loop=1000,
        ),

        load_demos=False,
        # demo_paths=[os.path.join(main_dir,"demos",demo)\
        # 			for demo in os.listdir(os.path.join(main_dir,"demos")) if f"{args.env_name}" in demo],
        demo_paths=[
            os.path.join(main_dir, "demos",
                         f"int_OneSwitch_sim_model_on_policy_1000_all_debug_1617655710048679887.npy"),

        ],
        # demo_path_proportions=[1]*9,
        pretrain_rf=False,
        pretrain=False,

        env_kwargs={'config': dict(
            env_name=args.env_name,
            step_limit=path_length,
            env_kwargs=dict(success_dist=.03, frame_skip=5, stochastic=True),
            oracle='sim_gaze_model',
            oracle_kwargs=dict(),
            action_type='disc_traj',
            smooth_alpha=.8,
            adapts=['high_dim_user', 'reward'],
            reward_type='user_input',
            input_in_obs=True,
            gaze_oracle_kwargs={'mode': 'train'},
        )},
    )
    search_space = {
        'freeze_encoder': [False],
        'seedid': [2000],
        'trainer_kwargs.temp': [1],
        'trainer_kwargs.min_q_weight': [10],
        'env_kwargs.config.oracle_kwargs.threshold': [.5],
        'env_kwargs.config.apply_projection': [False],
        'env_kwargs.config.input_penalty': [1],
        'demo_path_proportions': [[1000]],
        'trainer_kwargs.qf_lr': [5e-4],
        'trainer_kwargs.ground_truth': [True],
        'trainer_kwargs.add_ood_term': [-1],
        'expl_kwargs.eps': [0.1],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)


    def process_args(variant):
        variant['trainer_kwargs']['learning_rate'] = variant['trainer_kwargs'].pop('qf_lr')
        variant['qf_lr'] = variant['trainer_kwargs']['learning_rate']
        variant['env_kwargs']['config']['seedid'] = variant['seedid']
        if not args.use_ray:
            variant['render'] = args.no_render


    if args.use_ray:
        import ray
        from ray.util import ActorPool
        from itertools import cycle, count

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
                setup_logger(exp_prefix=args.exp_name, variant=variant, base_log_dir=save_path, exp_id=run_id, )
                experiment(variant)


        runners = [Runner.remote() for i in range(args.gpus * args.per_gpu)]
        runner_pool = ActorPool(runners)
        list(runner_pool.map(lambda a, v: a.run.remote(v), variants))
    else:
        import time

        current_time = time.time_ns()
        variant = variants[0]
        run_id = 0
        save_path = os.path.join(main_dir, 'logs')
        setup_logger(exp_prefix=args.exp_name, variant=variant, base_log_dir=save_path, exp_id=run_id)
        process_args(variant)
        experiment(variant)
