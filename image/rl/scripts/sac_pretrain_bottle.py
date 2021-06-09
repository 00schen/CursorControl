import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlp, VAE
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.sac.policies import ConcatTanhGaussianPolicy

from rl.policies import EncDecPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.trainers import EncDecSACTrainer
from rl.replay_buffers import ModdedReplayBuffer
from rl.scripts.run_util import run_exp

import os
from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
from functools import reduce
import operator


def experiment(variant):
    import torch as th

    env = default_overhead(variant['env_config'])
    env.seed(variant['seedid'])
    eval_config = variant['env_config'].copy()
    eval_env = default_overhead(eval_config)
    eval_env.seed(variant['seedid'] + 1)

    # qf takes in goal directly instead of latent, but same dim
    feat_dim = env.observation_space.low.size
    obs_dim = env.observation_space.low.size + reduce(operator.mul,
                                                      getattr(env.base_env, 'goal_set_shape', (0,)),
                                                      1) + sum(env.feature_sizes.values())
    action_dim = env.action_space.low.size
    M = variant["layer_size"]

    if not variant['from_pretrain']:
        qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        target_qf1 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        target_qf2 = ConcatMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M],
        )
        policy = ConcatTanhGaussianPolicy(
            obs_dim=feat_dim + variant['latent_size'],
            action_dim=action_dim,
            hidden_sizes=[M, M],
        )
        vae = VAE(input_size=sum(env.feature_sizes.values()) if env.env_name == 'OneSwitch' else obs_dim,
                  latent_size=variant['latent_size'],
                  encoder_hidden_sizes=[64],
                  decoder_hidden_sizes=[64]
                  ).to(ptu.device)
    else:
        file_name = os.path.join('util_models', variant['pretrain_path'])
        loaded = th.load(file_name)
        qf1 = loaded['trainer/qf1']
        qf2 = loaded['trainer/qf2']
        target_qf1 = loaded['trainer/target_qf1']
        target_qf2 = loaded['trainer/target_qf2']
        policy = loaded['trainer/policy']
        vae = loaded['trainer/vae']

    expl_policy = EncDecPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        vae=vae,
        incl_state=False,
        sample=False,
        deterministic=False
    )

    eval_policy = EncDecPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        vae=vae,
        incl_state=False,
        sample=False,
        deterministic=True
    )

    eval_path_collector = FullPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False
    )

    expl_path_collector = FullPathCollector(
        env,
        expl_policy,
        save_env_in_snapshot=False,
    )
    trainer = EncDecSACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vae=vae,
        latent_size=variant['latent_size'],
        **variant['trainer_kwargs']
    )
    replay_buffer = variant['buffer_type'](
        variant['replay_buffer_size'],
        env,
        sample_base=0,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_args']
    )
    algorithm.to(ptu.device)
    path_loader = SimplePathLoader(
        demo_path=variant['demo_paths'],
        demo_path_proportion=variant['demo_path_proportions'],
        replay_buffer=replay_buffer,
    )
    path_loader.load_demos()

    if variant.get('render', False):
        env.render('human')
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', )
    parser.add_argument('--exp_name', default='pretrain_sac')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    args, _ = parser.parse_known_args()
    main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200
    variant = dict(
        pretrain_path=f'{args.env_name}_params_s1_sac.pkl',
        latent_size=3,
        layer_size=256,
        algorithm_args=dict(
            num_epochs=int(1e6),
            num_eval_steps_per_epoch=0,
            eval_paths=False,
            # num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=path_length,
            batch_size=256,
            collect_new_paths=True,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            encoder_lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            sample=True,
        ),
        # demo_paths=[
        #     # os.path.join(main_dir, "demos", f"{args.env_name}_keyboard_on_policy_1_begin.npy"),
        #     os.path.join(main_dir, "demos", f"{args.env_name}_keyboard_on_policy_1_full1.npy"),
        # ]*500, # no latent
        demo_paths=[
            os.path.join(main_dir, "demos", f"{args.env_name}_model_on_policy_5000_full.npy"),
        ], # no latent
        env_config=dict(
            terminate_on_failure=False,
            env_name=args.env_name,
            step_limit=path_length,
            goal_noise_std=0,
            env_kwargs=dict(success_dist=.03, frame_skip=5, debug=False, num_targets=5, joint_in_state=False,),
            action_type='joint',
            smooth_alpha=1,
            factories=[],
            adapts=['goal', 'reward'],
            gaze_dim=128,
            state_type=0,
            reward_max=0,
            reward_min=-1,
            reward_type='part_sparse',
            reward_temp=1,
            reward_offset=-0.2
        )
    )
    search_space = {
        'seedid': [2000],
        'from_pretrain': [False],
        # 'demo_path_proportions': [[50]*500, ],
        'demo_path_proportions': [[5000], ],
        'trainer_kwargs.beta': [.01,.1],
        # 'trainer_kwargs.beta': [.01,],
        'algorithm_args.num_trains_per_train_loop': [100,1000],
        # 'algorithm_args.num_trains_per_train_loop': [1000,],
        'buffer_type': [ModdedReplayBuffer],
        'replay_buffer_size': [int(2e7)],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)


    def process_args(variant):
        variant['env_config']['seedid'] = variant['seedid']
        if not args.use_ray:
            variant['render'] = args.no_render


    args.process_args = process_args

    run_exp(experiment, variants, args)