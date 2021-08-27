import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import VAE
from rl.misc.calibration_rl_algorithm import BatchRLAlgorithm as TorchCalibrationRLAlgorithm

from rl.policies import CalibrationPolicy, EncDecPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.trainers import LatentEncDecSACTrainer
from rl.replay_buffers import ModdedReplayBuffer
from rl.scripts.run_util import run_exp

import os
from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
from torch import optim
from copy import deepcopy
from functools import reduce
import operator
import numpy as np


def experiment(variant):
    import torch as th

    expl_config = deepcopy(variant['env_config'])
    expl_config['factories'] += ['session']
    env = default_overhead(expl_config)

    eval_config = deepcopy(variant['env_config'])
    eval_config['gaze_path'] = eval_config['eval_gaze_path']
    eval_env = default_overhead(eval_config)

    M = variant["layer_size"]

    file_name = os.path.join('util_models', variant['pretrain_path'])
    loaded = th.load(file_name, map_location=ptu.device)

    feat_dim = env.observation_space.low.size + reduce(operator.mul,
                                                       getattr(env.base_env, 'goal_set_shape', (0,)), 1)
    obs_dim = feat_dim + sum(env.feature_sizes.values())

    vaes = []
    for _ in range(variant['n_encoders']):
        vaes.append(VAE(input_size=obs_dim if variant['incl_state'] else sum(env.feature_sizes.values()),
                        latent_size=variant['latent_size'],
                        encoder_hidden_sizes=[M] * variant['n_layers'],
                        decoder_hidden_sizes=[M] * variant['n_layers']
                        ).to(ptu.device))
    policy = loaded['trainer/policy']

    qf1 = loaded['trainer/qf1']
    qf2 = loaded['trainer/qf2']

    if 'trainer/vae' in loaded.keys():
        prev_vae = loaded['trainer/vae'].to(ptu.device)
    else:
        prev_vae = None

    optim_params = []
    for vae in vaes:
        optim_params += list(vae.encoder.parameters())
    if not variant['freeze_decoder']:
        optim_params += list(policy.parameters())

    optimizer = optim.Adam(
        optim_params,
        lr=variant['lr'],
    )

    expl_policy = EncDecPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        vaes=vaes,
        incl_state=variant['incl_state'],
        sample=variant['sample'],
        deterministic=True,
        latent_size=variant['latent_size'],
        random_latent=variant.get('random_latent', False),
        window=variant['window'],
        prev_vae=prev_vae if variant['trainer_kwargs']['objective'] == 'goal' else None
    )

    eval_policy = EncDecPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        vaes=vaes,
        incl_state=variant['incl_state'],
        sample=variant['sample'],
        deterministic=True,
        latent_size=variant['latent_size'],
        window=variant['window'],
        prev_vae=prev_vae if variant['trainer_kwargs']['objective'] == 'goal' else None
    )

    eval_path_collector = FullPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False
    )

    calibration_policy = CalibrationPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        env=env,
        vaes=vaes,
        prev_vae=prev_vae,
        incl_state=variant['trainer_kwargs']['prev_incl_state']
    )
    expl_path_collector = FullPathCollector(
        env,
        expl_policy,
        save_env_in_snapshot=False,
        real_user=variant['real_user']
    )
    calibration_path_collector = FullPathCollector(
        env,
        calibration_policy,
        save_env_in_snapshot=False,
        real_user=variant['real_user']
    )
    replay_buffer = ModdedReplayBuffer(
        variant['replay_buffer_size'],
        env,
        sample_base=0,
        latent_size=variant['latent_size'],
        store_latents=True,
        window_size=variant['window']
    )
    trainer = LatentEncDecSACTrainer(
        vaes=vaes,
        prev_vae=prev_vae,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        optimizer=optimizer,
        latent_size=variant['latent_size'],
        feature_keys=list(env.feature_sizes.keys()),
        incl_state=variant['incl_state'],
        **variant['trainer_kwargs']
    )

    if variant['balance_calibration']:
        calibration_buffer = ModdedReplayBuffer(
            variant['replay_buffer_size'],
            env,
            sample_base=0,
            latent_size=variant['latent_size'],
            store_latents=True,
            window_size=variant['window']
        )
    else:
        calibration_buffer = None

    if variant['real_user']:
        variant['algorithm_args']['eval_paths'] = False

    algorithm = TorchCalibrationRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        calibration_data_collector=calibration_path_collector,
        calibration_buffer=calibration_buffer,
        real_user=variant['real_user'],
        **variant['algorithm_args']
    )
    algorithm.to(ptu.device)

    if variant.get('render', False):
        env.render('human')
    algorithm.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='BlockPush')
    parser.add_argument('--exp_name', default='experiments')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    parser.add_argument('--mode', default='default', type=str)
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--det', action='store_true')
    parser.add_argument('--pre_det', action='store_true')
    parser.add_argument('--no_failures', action='store_true')
    parser.add_argument('--rand_latent', action='store_true')
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--objective', default='normal_kl', type=str)
    parser.add_argument('--prev_incl_state', action='store_true')

    args, _ = parser.parse_known_args()
    main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200

    pretrain_path = f'{args.env_name}_params_s1_sac_det'
    if args.pre_det:
        pretrain_path += '_det_enc'
    pretrain_path += '.pkl'

    default_variant = dict(
        mode=args.mode,
        incl_state=True,
        sample=False,
        random_latent=args.rand_latent,
        real_user=not args.sim,
        pretrain_path=pretrain_path,
        n_layers=1,
        n_encoders=1,
        latent_size=4,
        layer_size=64,
        freeze_decoder=True,
        balance_calibration=True,
        replay_buffer_size=int(1e3 * path_length), # calibration is short
        keep_calibration_data=True,
        trainer_kwargs=dict(
            sample=not args.det,
            beta=0 if args.det else 1e-4,
            objective=args.objective,
            grad_norm_clip=None,
            prev_incl_state=args.prev_incl_state
        ),
        algorithm_args=dict(
            batch_size=256,
            max_path_length=path_length,
            num_epochs=args.epochs,
            num_eval_steps_per_epoch=1000,
            num_train_loops_per_epoch=1,
            collect_new_paths=True,
            pretrain_steps=1000,
            max_failures=3,
            eval_paths=False,
            relabel_failures=not args.no_failures,
            curriculum=args.curriculum
        ),

        env_config=dict(
            env_name=args.env_name,
            goal_noise_std=0.1,
            terminate_on_failure=True,
            env_kwargs=dict(frame_skip=5, debug=False,
                            stochastic=False, num_targets=4, min_error_threshold=np.pi / 8,
                            use_rand_init_angle=False,
                            term_cond='auto',
                            curriculum=args.curriculum,
                            always_reset=True),
            action_type='joint',
            smooth_alpha=1,
            factories=[],
            # adapts=['goal', 'sim_target'],
            feature='goal',
            gaze_dim=128,
            gaze_path=f'{args.env_name}_gaze_data_train.h5',
            eval_gaze_path=f'{args.env_name}_gaze_data_eval.h5',
        )
    )
    variants = []

    search_space = {
        'algorithm_args.trajs_per_index': [5],
        'lr': [5e-4],
        'algorithm_args.num_trains_per_train_loop': [100],
        'seedid': [0,1,2,3,4],
        'env_config.adapts': [['goal', 'sim_keyboard'],],
        'env_config.mode': ['oracle'],
        'env_config.keyboard_size': [14],
        # 'env_config.env_kwargs.always_reset': [True, False],
        # 'use_np': [True, False]
        'window': [1, 5, 10, 20]
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=default_variant,
    )
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    # search_space = {
    #     'algorithm_args.trajs_per_index': [5],
    #     'lr': [5e-4],
    #     'algorithm_args.num_trains_per_train_loop': [100],
    #     'seedid': [0,1,2,3,4],
    #     'env_config.adapts': [['goal', 'sim_target'],],
    #     # 'env_config.mode': ['block', 'tool'],
    #     # 'env_config.env_kwargs.always_reset': [True, False],
    #     # 'use_np': [True, False]
    #     'window': [1, 5, 10, 20]
    # }

    # sweeper = hyp.DeterministicHyperparameterSweeper(
    #     search_space, default_parameters=default_variant,
    # )
    # for variant in sweeper.iterate_hyperparameters():
    #     variants.append(variant)

    def process_args(variant):
        variant['env_config']['seedid'] = variant['seedid']
        variant['algorithm_args']['seedid'] = variant['seedid']

        if not args.use_ray:
            variant['render'] = args.no_render

        variant['trainer_kwargs']['window_size'] = variant['window']

        mode_dict = {'default': {'calibrate_split': False,
                                'calibration_indices': None},
                    'no_online': {'calibrate_split': False,
                                'calibration_indices': None,
                                'num_trains_per_train_loop': 0},
                    }[variant['mode']]

        variant['algorithm_args'].update(mode_dict)

        if variant['real_user']:
            variant['env_config']['adapts'] = ['goal', 'real_user']

        if variant['trainer_kwargs']['objective'] == 'awr':
            variant['algorithm_args']['relabel_failures'] = False

    args.process_args = process_args

    run_exp(experiment, variants, args)
