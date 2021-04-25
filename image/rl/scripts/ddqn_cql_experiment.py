import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlp, MlpPolicy, MlpGazePolicy, MlpVQVAEGazePolicy
from rlkit.torch.networks import Clamp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.env_relabeling_replay_buffer import EnvRelabelingReplayBuffer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import BoltzmannPolicy, OverridePolicy, ComparisonMergePolicy, ArgmaxPolicy, UserInputPolicy
from rl.path_collectors import FullPathCollector
from rl.env_wrapper import default_overhead
from rl.simple_path_loader import SimplePathLoader
from rl.trainers import DDQNCQLTrainer, DDQNVQCQLTrainer

import os
import gtimer as gt
from pathlib import Path
from rlkit.launchers.launcher_util import setup_logger, reset_execution_environment
import rlkit.util.hyperparameter as hyp
import argparse
import torch as th
from torch.nn import functional as F
import copy
import pickle as pkl


def experiment(variant):
    gaze_vae = None
    if variant['gaze_vae_path'] is not None:
        gaze_vae = th.load(variant['gaze_vae_path'], map_location=th.device("cpu"))

    encoder = None
    decoder = None
    rew_classifier = None
    if variant['pretrain_file_path'] is not None:
        load = th.load(variant['pretrain_file_path'], map_location=th.device("cpu"))['trainer/qf']
        if variant['load_encoder']:
            encoder = load.gaze_vae
        if variant['load_decoder']:
            decoder = load.decoder
        if variant['load_rew_classifier']:
            rew_classifier = load.rew_classifier
        if hasattr(load, 'rew_classifier'):
            variant['env_kwargs']['config']['rew_fn'] = load.rew_classifier

    env_config = copy.deepcopy(variant['env_kwargs']['config'])
    if variant['trainer_kwargs']['latent_train']:
        env_config['oracle'] = 'sim_latent_model'
        env_config['reward_type'] = 'rew_fn'
        env_config['gaze_oracle_kwargs']['gazes_path'] = variant['gazes_path']
        env_config['gaze_oracle_kwargs']['encoder'] = encoder

    env = default_overhead(env_config)
    env.seed(variant['seedid'])

    eval_env_config = copy.deepcopy(variant['env_kwargs']['config'])
    eval_env_config['seedid'] += 100
    eval_env_config['gaze_oracle_kwargs']['mode'] = 'eval'

    eval_env = default_overhead(eval_env_config)
    eval_env.seed(variant['seedid'] + 100)

    if not variant['from_pretrain']:
        obs_dim = eval_env.observation_space.low.size
        action_dim = eval_env.action_space.low.size
        gaze_dim = 128 if 'gaze' in eval_env_config['oracle'] else variant['embedding_dim'] * variant['n_latents']
        M = variant["layer_size"]
        qf = MlpGazePolicy(
            input_size=obs_dim,
            output_size=action_dim,
            encoder_hidden_sizes=[64],
            decoder_hidden_sizes=[M, M, M, M],
            hidden_activation=F.relu,
            layer_norm=True,
            gaze_vae=gaze_vae,
            gaze_dim=gaze_dim,
            embedding_dim=variant['embedding_dim'],
            decoder=decoder,
            rew_classifier=rew_classifier,
            # n_embed_per_latent=variant['n_embed_per_latent'],
            # n_latents=variant['n_latents']
        )
        target_qf = MlpGazePolicy(
            input_size=obs_dim,
            output_size=action_dim,
            encoder_hidden_sizes=[64],
            decoder_hidden_sizes=[M, M, M, M],
            hidden_activation=F.relu,
            layer_norm=True,
            gaze_vae=copy.deepcopy(gaze_vae),
            gaze_dim=gaze_dim,
            embedding_dim=variant['embedding_dim'],
            decoder=copy.deepcopy(decoder),
            rew_classifier=copy.deepcopy(rew_classifier),
            # n_embed_per_latent=variant['n_embed_per_latent'],
            # n_latents=variant['n_latents']
        )
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
        eval_env,
        eval_policy,
        save_env_in_snapshot=False
    )
    if not variant['exploration_argmax']:
        expl_policy = BoltzmannPolicy(
            qf,
            logit_scale=variant['expl_kwargs']['logit_scale'])
    else:
        expl_policy = ArgmaxPolicy(
            qf, eps=variant['expl_kwargs']['eps'], skip_encoder=variant['trainer_kwargs']['latent_train']
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

    buffer_type = EnvRelabelingReplayBuffer if variant['her'] else EnvReplayBuffer

    online_buffer = buffer_type(
         variant['replay_buffer_size'],
         env,
    )
    offline_buffer = buffer_type(
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

    algorithm.train()

    if hasattr(env.oracle, 'gazes'):
        gaze_save_path = os.path.join(logger.get_snapshot_dir(), 'gazes.pkl')
        with open(gaze_save_path, 'wb') as f:
            pkl.dump(env.oracle.gazes, f)

# online params
# from pretrain = False
# pretrain file path = pretrain
# sample = False
# latent train = False
# rew class weight = 1
# train encoder on rew class = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', )
    parser.add_argument('--exp_name', default='vq_her')
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
        her=True,
        from_pretrain=False,
        pretrain_file_path=os.path.join(main_dir, 'logs', 'her/her_2021_04_23_21_15_39_0000--s-0',
                                        'params.pkl'),
        # pretrain_file_path=os.path.join(main_dir, 'logs', 'pretrain/pretrain_2021_04_18_10_23_44_0000--s-0',
                                        # 'params.pkl'),
        load_decoder=False,
        load_encoder=False,
        load_rew_classifier=False,
        gaze_vae_path=None,#os.path.join(main_dir, 'logs', 'gaze_vae', '2021-04-08--18-38-06.pkl'),
        gazes_path=os.path.join(main_dir, 'logs', 'online/online_2021_04_18_12_54_28_0000--s-0', 'gazes.pkl'),
        layer_size=128,
        embedding_dim=3,
        n_latents=1,
        n_embed_per_latent=50,
        exploration_argmax=True,
        exploration_strategy='',
        expl_kwargs=dict(
            logit_scale=10,
        ),
        replay_buffer_size=100000,
        trainer_kwargs=dict(
            soft_target_tau=1e-2,
            target_update_period=1,
            qf_criterion=None,
            discount=0.99,
            reward_scale=1.0,
            beta=0.1,
            rew_class_weight=1,
            sample=True,
            latent_train=False,
            train_encoder_on_rew_class=False,
            freeze_decoder=False,
            train_qf_head=False,
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
            # batch_size=128,
            # max_path_length=path_length,
            # eval_path_length=path_length,
            # num_epochs=100,
            # num_eval_steps_per_epoch=1,
            # num_trains_per_train_loop=5,
            # num_expl_steps_per_train_loop=1,
            # min_num_steps_before_training=1000,
            # num_train_loops_per_epoch=1,
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
            oracle='sim_goal_model',
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
        'trainer_kwargs.min_q_weight': [0],
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
