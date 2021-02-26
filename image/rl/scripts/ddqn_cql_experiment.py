import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp, ConcatMlp, QfGazeMlp, QfMlp, MlpPolicy
from rlkit.torch.networks import Clamp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
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


def experiment(variant):
    env = default_overhead(variant['env_kwargs']['config'])
    env.seed(variant['seedid'])

    if not variant['from_pretrain']:
        obs_dim = env.observation_space.low.size
        action_dim = env.action_space.low.size
        M = variant["layer_size"]
        upper_q = 1 # variant['env_kwargs']['config']['reward_max'] # * variant['env_kwargs']['config']['step_limit']
        lower_q = -1 #variant['env_kwargs']['config']['reward_min'] * variant['env_kwargs']['config']['step_limit']
        im_path = os.path.join(main_dir, 'logs', 'bc-gaze', 'bc_gaze_2021_02_23_16_05_08_0000--s-0', 'pretrain.pkl')
        im_policy = th.load(im_path,map_location=th.device("cpu"))
        qf = QfGazeMlp(
            input_size=obs_dim,
            action_size=action_dim,
            hidden_sizes=[M, M, M, M],
            hidden_activation=F.relu,
            layer_norm=True,
            # output_activation=output,
            reward_min=-1 * variant['env_kwargs']['config']['step_limit'],
            gaze_encoder=im_policy.encoder,
            gaze_dim=im_policy.gaze_dim,
            embedding_dim=im_policy.embedding_dim,
        )
        target_qf = QfGazeMlp(
            input_size=obs_dim,
            action_size=action_dim,
            hidden_sizes=[M, M, M, M],
            hidden_activation=F.relu,
            layer_norm=True,
            # output_activation=output,
            reward_min=-1 * variant['env_kwargs']['config']['step_limit'],
            gaze_encoder=im_policy.encoder,
            gaze_dim=im_policy.gaze_dim,
            embedding_dim=im_policy.embedding_dim,
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
        qf = th.load(pretrain_file_path, map_location=th.device("cpu"))['qf']
        target_qf = th.load(pretrain_file_path, map_location=th.device("cpu"))['target_qf']
        rf = th.load(pretrain_file_path, map_location=th.device("cpu"))['rf']
    for param in qf.gaze_encoder.parameters():
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
    trainer = QRDDQNCQLTrainer(
        qf=qf,
        target_qf=target_qf,
        rf=rf,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        env,
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
    # if variant['pretrain_rf']:
    # 	path_loader = SimplePathLoader(
    # 		demo_path=variant['demo_paths'],
    # 		demo_path_proportion=[1,1],
    # 		replay_buffer=replay_buffer,
    # 	)
    # 	path_loader.load_demos()
    # 	from tqdm import tqdm
    # 	for _ in tqdm(range(int(1e5)),miniters=10,mininterval=10):
    # 		train_data = replay_buffer.random_batch(variant['algorithm_args']['batch_size'])
    # 		trainer.pretrain_rf(train_data)
    # 	algorithm.replay_buffer = None
    # 	del replay_buffer
    # 	replay_buffer = EnvReplayBuffer(
    # 		variant['replay_buffer_size'],
    # 		env,
    # 	)
    # 	algorithm.replay_buffer = replay_buffer
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
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'pretrain.csv', relative_to_snapshot_dir=True, )
        # data = h5py.File('image/rl/gaze_capture/gaze_data.h5', 'r')
        # features = []
        # labels = []
        # for i, key in enumerate(data.keys()):
        # 	features.extend(data[key][()])
        # 	for j in range(len(data[key][()])):
        # 		labels.append(i)
        # batch = {'features': np.array(features), 'labels': np.array(labels)}
        # torch_batch = np_to_pytorch_batch(batch)
        # optimizer = optim.Adam(
        # 	vq_vae.parameters(),
        # 	lr=1e-3,
        # )
        # for i in range(1000):
        # 	optimizer.zero_grad()
        # 	vq_loss, pred, perplexity = vq_vae(torch_batch['features'])
        # 	pred_loss = F.cross_entropy(pred, torch_batch['labels'].long())
        # 	loss = pred_loss + vq_loss
        # 	loss.backward()
        # 	optimizer.step()
        # 	print(vq_loss, pred_loss)
        # ptu.copy_model_params_from_to(qf, target_qf)

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

    # for g in trainer.qf_optimizer.param_groups:
    #     g['lr'] = 5e-4

    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', )
    parser.add_argument('--exp_name', default='sparse_gaze')
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
        from_pretrain=True,
        pretrain_file_path=os.path.join(main_dir, 'logs', 'sparse-gaze', 'sparse_gaze_2021_02_24_09_01_35_0000--s-0',
                                        'pretrain.pkl'),
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
        ),
        algorithm_args=dict(
            batch_size=64,
            max_path_length=path_length,
            eval_path_length=path_length,
            num_epochs=250,
            num_eval_steps_per_epoch=path_length * 3,
            num_expl_steps_per_train_loop=1,
            num_train_loops_per_epoch=10,
            # num_trains_per_train_loop=50,
            # min_num_steps_before_training=1000,
        ),
        bc_args=dict(
            batch_size=64,
            max_path_length=path_length,
            num_epochs=250,
            num_eval_steps_per_epoch=path_length * 3,
            num_expl_steps_per_train_loop=0,
            collect_new_paths=False,
            num_trains_per_train_loop=1000,
        ),

        load_demos=True,
        # demo_paths=[os.path.join(main_dir,"demos",demo)\
        # 			for demo in os.listdir(os.path.join(main_dir,"demos")) if f"{args.env_name}" in demo],
        demo_paths=[
            os.path.join(main_dir, "demos",
                         f"int_OneSwitch_sim_gaze_on_policy_100_all_debug_1614124074734074185.npy"),
            os.path.join(main_dir, "demos",
                         f"BC_OneSwitch_sim_gaze_model_on_policy_1000_all_debug_1614144229823834917.npy")


        ],
        # demo_path_proportions=[1]*9,
        pretrain_rf=False,
        pretrain=False,

        env_kwargs={'config': dict(
            env_name=args.env_name,
            step_limit=path_length,
            env_kwargs=dict(success_dist=.03, frame_skip=5),
            # env_kwargs=dict(path_length=path_length,frame_skip=5),

            oracle='sim_gaze_model',
            oracle_kwargs=dict(),
            action_type='disc_traj',
            smooth_alpha=.8,

            adapts=['high_dim_user', 'reward'],
            space=0,
            num_obs=10,
            num_nonnoop=0,
            # reward_max=0,
            reward_min=-1,
            # input_penalty=1,
            reward_type='user_penalty',
            input_in_obs=True,
            gaze_oracle_kwargs={'mode': 'rl'},
        )},
    )
    search_space = {
        'seedid': [2000],
        'trainer_kwargs.temp': [1],
        'trainer_kwargs.min_q_weight': [10],
        'env_kwargs.config.oracle_kwargs.threshold': [.5],
        'env_kwargs.config.apply_projection': [False],
        'env_kwargs.config.input_penalty': [1],
        'env_kwargs.config.reward_max': [0],
        # 'demo_path_proportions':[[int(1e4),int(1e4)],[int(1e4),0],[int(5e3),0]],
        'demo_path_proportions': [[1000, 1000]],
        # 'demo_path_proportions':[[25,25],[50,50],[100,100],[250,250]],
        'trainer_kwargs.qf_lr': [5e-5],
        'algorithm_args.num_trains_per_train_loop': [100],
        # 'trainer_kwargs.reward_update_period':[10],
        'trainer_kwargs.ground_truth': [True],
        'trainer_kwargs.add_ood_term': [-1],
        'expl_kwargs.eps': [0],
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
