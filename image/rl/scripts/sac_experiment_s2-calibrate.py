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

    # separate calibration env for seeding purposes

    eval_config = deepcopy(variant['env_config'])
    eval_config['gaze_path'] = eval_config['eval_gaze_path']
    eval_env = default_overhead(eval_config)

    M = variant["layer_size"]

    file_name = os.path.join('image','util_models', variant['pretrain_path'])
    loaded = th.load(file_name, map_location=ptu.device)

    obs_dim = env.observation_space.low.size + reduce(operator.mul,
                                                      getattr(env.base_env, 'goal_set_shape', (0,)),
                                                      1)

    vae = VAE(input_size=sum(env.feature_sizes.values()) + obs_dim,
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
        latent_size=variant['latent_size']
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
            latent_size=variant['latent_size']
        )

    if variant['real_user']:
        variant['algorithm_args']['eval_paths'] = False

    if env.env_name == 'Bottle':
        variant['algorithm_args']['calibration_indices'] = [0,3]
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
    parser.add_argument('--exp_name', default='calibrate_sac_test')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    parser.add_argument('--mode', default='default', type=str)
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--epochs', default=50, type=int)
    args, _ = parser.parse_known_args()
    main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200
    default_variant = dict(
        # mode=args.mode,
        real_user=not args.sim,
        pretrain_path=f'{args.env_name}_params_s1_sac.pkl',
        latent_size=3,
        layer_size=64,
        replay_buffer_size=int(1e4 * path_length),
        keep_calibration_data=True,
        trainer_kwargs=dict(
            beta=0.01,
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
        ),

        env_config=dict(
            env_name=args.env_name,
            goal_noise_std=0.05,
            terminate_on_failure=True,
            env_kwargs=dict(step_limit=path_length, success_dist=.03, frame_skip=5, debug=False, num_targets=5,
                            target_indices=[1, 2, 3]),

            action_type='joint',
            smooth_alpha=1,

            factories=[],
            adapts=['goal', 'reward'],
            gaze_dim=128,
            state_type=0,
            reward_max=0,
            reward_min=-1,
            reward_temp=1,
            reward_offset=-0.1,
            reward_type='sparse',
            gaze_path=f'{args.env_name}_gaze_data_train.h5',
            eval_gaze_path=f'{args.env_name}_gaze_data_eval.h5'
        )
    )
    variants = []

    search_space = {
        # 'seedid': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
        'n_layers': [1],
        'algorithm_args.trajs_per_index': [2],
        'lr': [5e-4],
        'trainer_kwargs.sample': [True],
        # 'algorithm_args.calibrate_split': [False],
        # 'algorithm_args.calibration_indices': [[1, 2, 3]],
        'algorithm_args.relabel_failures': [True],
        'algorithm_args.num_trains_per_train_loop': [100],
        # 'algorithm_args.seedid': [0],
        'trainer_kwargs.objective': ['kl', 'joint', 'awr'],
        'mode': ['default', 'no_online', 'shift', 'no_right'],
        'algorithm_args.seedid': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
        'freeze_decoder': [True],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=default_variant,
    )
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    if args.env_name == 'Bottle':
        variants[0]['algorithm_args']['calibration_indices'] = [0,3]

    # search_space = {
    #     'seedid': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
    #     'n_layers': [1],
    #     'algorithm_args.trajs_per_index': [3],
    #     'lr': [5e-4],
    #     'trainer_kwargs.sample': [True],
    #     'algorithm_args.calibrate_split': [False,],
    #     'algorithm_args.calibration_indices': [[0], [4]],
    #     'algorithm_args.max_path_length': [path_length, 5*path_length, 10*path_length],
    #     'algorithm_args.max_failures': [1],
    #     'freeze_decoder': [True],
    #     'trainer_kwargs.use_supervised': ['calibrate_kl'],
    # }
    # sweeper = hyp.DeterministicHyperparameterSweeper(
    #     search_space, default_parameters=default_variant,
    # )
    # for variant in sweeper.iterate_hyperparameters():
    #     variants.append(variant)

    def process_args(variant):
        if not args.use_ray:
            variant['render'] = args.no_render

        mode_dict = {'default': {'calibrate_split': False,
                                 'calibration_indices': [1, 2, 3]},
                     'no_online': {'calibrate_split': False,
                                   'calibration_indices': [1, 2, 3],
                                   'num_trains_per_train_loop': 0},
                     'shift': {'calibrate_split': True,
                               'calibration_indices': [1, 2, 3]},
                     'no_right': {'calibrate_split': False,
                                  'calibration_indices': [2, 3]},
                     'overcalibrate': {'calibrate_split': False,
                                       'calibration_indices': [0, 1, 2, 3, 4]}}[variant['mode']]

        variant['algorithm_args'].update(mode_dict)

        if variant['real_user']:
            variant['env_config']['adapts'].insert(1, 'real_gaze')

        if variant['trainer_kwargs']['objective'] == 'awr':
            variant['algorithm_args']['relabel_failures'] = False

    args.process_args = process_args

    run_exp(experiment, variants, args)