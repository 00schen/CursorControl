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


def experiment(variant):
    import torch as th

    expl_config = deepcopy(variant['env_config'])
    expl_config['factories'] += ['session']
    env = default_overhead(expl_config)

    eval_config = deepcopy(variant['env_config'])
    eval_config['gaze_path'] = eval_config['eval_gaze_path']
    eval_env = default_overhead(eval_config)

    M = variant["layer_size"]

    file_name = os.path.join('image', 'util_models', variant['pretrain_path'])
    loaded = th.load(file_name, map_location=ptu.device)

    feat_dim = env.observation_space.low.size + reduce(operator.mul,
                                                       getattr(env.base_env, 'goal_set_shape', (0,)), 1)
    obs_dim = feat_dim + sum(env.feature_sizes.values())

    vae = VAE(input_size=obs_dim,
              latent_size=variant['latent_size'],
              encoder_hidden_sizes=[M] * variant['n_layers'],
              decoder_hidden_sizes=[M] * variant['n_layers']
              ).to(ptu.device)
    policy = loaded['trainer/policy']

    qf1 = loaded['trainer/qf1']
    qf2 = loaded['trainer/qf2']

    if 'trainer/vae' in loaded.keys():
        prev_vae = loaded['trainer/vae'].to(ptu.device)
    else:
        prev_vae = None

    optim_params = list(vae.encoder.parameters())
    if not variant['freeze_decoder']:
        optim_params += list(policy.parameters())

    optimizer = optim.Adam(
        optim_params,
        lr=variant['lr'],
    )

    expl_policy = EncDecPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        vae=vae,
        incl_state=True,
        sample=variant['trainer_kwargs']['sample'],
        deterministic=True,
        latent_size=variant['latent_size'],
        random_latent=variant.get('random_latent', False)
    )

    eval_policy = EncDecPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        vae=vae,
        incl_state=True,
        sample=variant['trainer_kwargs']['sample'],
        deterministic=True,
        latent_size=variant['latent_size'],
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
        vae=vae,
        prev_vae=prev_vae,
        incl_state=False
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
        store_latents=True
    )
    trainer = LatentEncDecSACTrainer(
        vae=vae,
        prev_vae=prev_vae,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        optimizer=optimizer,
        latent_size=variant['latent_size'],
        feature_keys=list(env.feature_sizes.keys()),
        **variant['trainer_kwargs']
    )

    if variant['keep_calibration_data']:
        calibration_buffer = replay_buffer
    else:
        calibration_buffer = ModdedReplayBuffer(
            variant['replay_buffer_size'],
            env,
            sample_base=0,
            latent_size=variant['latent_size'],
            store_latents=True
        )

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
    parser.add_argument('--env_name', )
    parser.add_argument('--exp_name', default='experiments')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    parser.add_argument('--mode', default='default', type=str)
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--det', action='store_true')
    parser.add_argument('--pre_det', action='store_true')
    parser.add_argument('--no_failures', action='store_true')
    parser.add_argument('--latent_reg', action='store_true')
    parser.add_argument('--rand_latent', action='store_true')
    args, _ = parser.parse_known_args()
    main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200
    target_indices = [1, 2, 3] if args.env_name == 'OneSwitch' else None
    goal_noise_std = 0.1 if args.env_name == 'OneSwitch' else 0.15
    pretrain_path = f'{args.env_name}_params_s1_sac'
    if args.pre_det:
        pretrain_path += '_det_enc'
    pretrain_path += '.pkl'
    default_variant = dict(
        mode=args.mode,
        real_user=not args.sim,
        pretrain_path=pretrain_path,
        latent_size=3,
        layer_size=64,
        replay_buffer_size=int(1e4 * path_length),
        keep_calibration_data=True,
        trainer_kwargs=dict(
            sample=not args.det,
            beta=0 if args.det else 0.01,
            objective='awr' if args.latent_reg else 'kl',
            grad_norm_clip=None
        ),
        algorithm_args=dict(
            batch_size=256,
            max_path_length=path_length,
            num_epochs=args.epochs,
            num_eval_steps_per_epoch=1000,
            num_train_loops_per_epoch=1,
            collect_new_paths=True,
            pretrain_steps=1000,
            max_failures=5,
            eval_paths=False,
            relabel_failures=not args.no_failures
        ),

        env_config=dict(
            env_name=args.env_name,
            goal_noise_std=goal_noise_std,
            terminate_on_failure=True,
            env_kwargs=dict(step_limit=path_length, frame_skip=5, debug=False, target_indices=target_indices),
            action_type='joint',
            smooth_alpha=1,
            factories=[],
            adapts=['goal'],
            gaze_dim=128,
            gaze_path=f'{args.env_name}_gaze_data_train.h5',
            eval_gaze_path=f'{args.env_name}_gaze_data_eval.h5'
        )
    )
    variants = []

    search_space = {
        'n_layers': [1],
        'algorithm_args.trajs_per_index': [2],
        'lr': [5e-4],
        'trainer_kwargs.sample': [True],
        'algorithm_args.num_trains_per_train_loop': [100],
        'trainer_kwargs.objective': ['kl'],
        'env_config.feature': [None],
        'seedid': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'freeze_decoder': [True],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=default_variant,
    )
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    def process_args(variant):
        variant['env_config']['seedid'] = variant['seedid']
        variant['algorithm_args']['seedid'] = variant['seedid']

        if not args.use_ray:
            variant['render'] = args.no_render

        mode_dict = {'OneSwitch':
                         {'default': {'calibrate_split': False,
                                      'calibration_indices': [1, 2, 3]},
                          'no_online': {'calibrate_split': False,
                                        'calibration_indices': [1, 2, 3],
                                        'num_trains_per_train_loop': 0},
                          'shift': {'calibrate_split': True,
                                    'calibration_indices': [1, 2, 3]},
                          'no_right': {'calibrate_split': False,
                                       'calibration_indices': [2, 3]},
                          'overcalibrate': {'calibrate_split': False,
                                            'calibration_indices': [0, 1, 2, 3, 4]}
                          },
                     'Bottle':
                         {'default': {'calibrate_split': False,
                                      'calibration_indices': [0, 1, 2, 3]},
                          'no_online': {'calibrate_split': False,
                                        'calibration_indices': [0, 1, 2, 3],
                                        'num_trains_per_train_loop': 0},
                          'shift': {'calibrate_split': True,
                                    'calibration_indices': [0, 1, 2, 3]},
                          'no_door': {'calibrate_split': False,
                                      'calibration_indices': [1, 2]},
                          'with_door': {'calibrate_split': False,
                                        'calibration_indices': [0, 3]}

                          }
                     }[variant['env_config']['env_name']][variant['mode']]

        variant['algorithm_args'].update(mode_dict)

        target = 'real_gaze' if variant['real_user'] else 'sim_target'
        variant['env_config']['adapts'].append(target)

        if variant['trainer_kwargs']['objective'] == 'awr':
            variant['algorithm_args']['relabel_failures'] = False


    args.process_args = process_args

    run_exp(experiment, variants, args)
