import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import VAE
from rl.misc.calibration_rl_algorithm import BatchRLAlgorithm as TorchCalibrationRLAlgorithm

from rl.policies import EncDecQfPolicy, CalibrationDQNPolicy, KeyboardPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.trainers import LatentEncDecCQLTrainer
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

    # gaze_type = 'real_gaze' if variant['real_user'] else ['static_gaze']
    # variant['env_config']['adapts'].insert(1, gaze_type)

    expl_config = deepcopy(variant['env_config'])
    if 'calibrate' in variant['trainer_kwargs']['use_supervised']:
        expl_config['factories'] += ['session']
    env = default_overhead(expl_config)
    env.seed(variant['seedid'])
    eval_config = deepcopy(variant['env_config'])
    eval_config['gaze_path'] = eval_config['eval_gaze_path']
    eval_env = default_overhead(eval_config)
    eval_env.seed(variant['seedid'] + 1)

    M = variant["layer_size"]

    file_name = os.path.join('image/util_models', variant['pretrain_path'])
    loaded = th.load(file_name)

    obs_dim = env.observation_space.low.size + reduce(operator.mul,
                                                      getattr(env.base_env, 'goal_set_shape', (0,)),
                                                      1)

    vae = VAE(input_size=sum(env.feature_sizes.values()) + obs_dim,
              latent_size=variant['latent_size'],
              encoder_hidden_sizes=[M] * variant['n_layers'],
              decoder_hidden_sizes=[M] * variant['n_layers']
              ).to(ptu.device)

    qf = loaded['trainer/qf']
    target_qf = loaded['trainer/target_qf']
    if 'trainer/vae' in loaded.keys():
        prev_vae = loaded['trainer/vae'].to(ptu.device)
    else:
        prev_vae = None

    optim_params = list(vae.encoder.parameters())
    if not variant['freeze_decoder']:
        optim_params += list(qf.parameters())

    optimizer = optim.Adam(
        optim_params,
        lr=variant['lr'],
    )

    eval_policy = EncDecQfPolicy(
        qf,
        list(env.feature_sizes.keys()),
        vae=vae,
        sample=variant['trainer_kwargs']['sample'],
        latent_size=variant['latent_size'],
        eps=0,
        logit_scale=-1,
        incl_state=True
    )
    eval_path_collector = FullPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False
    )
    expl_policy = EncDecQfPolicy(
        qf=qf,
        features_keys=list(env.feature_sizes.keys()),
        vae=vae,
        sample=variant['trainer_kwargs']['sample'],
        latent_size=variant['latent_size'],
        eps=0,
        logit_scale=-1,
        incl_state=True
    )
    calibration_policy = CalibrationDQNPolicy(
        qf=qf,
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
    trainer = LatentEncDecCQLTrainer(
        vae=vae,
        prev_vae=prev_vae,
        qf=qf,
        target_qf=target_qf,
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

    # no eval paths with real user
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
        **variant['algorithm_args']
    )
    algorithm.to(ptu.device)

    if variant.get('render', False):
        env.render('human')
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', )
    parser.add_argument('--exp_name', default='calibrate_dqn')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    args, _ = parser.parse_known_args()
    main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200
    variant = dict(
        real_user=False,
        pretrain_path=f'{args.env_name}_params_s1_dqn.pkl',
        latent_size=3,
        layer_size=64,
        replay_buffer_size=int(1e4 * path_length),
        keep_calibration_data=True,
        trainer_kwargs=dict(
            temp=10,
            beta=0.01,
            grad_norm_clip=None
        ),
        algorithm_args=dict(
            batch_size=256,
            max_path_length=path_length,
            num_epochs=100,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1,
            num_train_loops_per_epoch=1,
            collect_new_paths=True,
            num_trains_per_train_loop=5,
            pretrain_steps=1000,
            max_failures=5,
            eval_paths=False
        ),

        env_config=dict(
            env_name=args.env_name,
            goal_noise_std=0.1,
            terminate_on_failure=True,
            step_limit=path_length,
            env_kwargs=dict(success_dist=.03, frame_skip=5, debug=False, num_targets=5, joint_in_state=False,
                            target_indices=[0, 2, 4]),

            action_type='disc_traj',
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
        'n_layers': [1],
        'algorithm_args.trajs_per_index': [3],
        'lr': [5e-4],
        'trainer_kwargs.sample': [True],
        'algorithm_args.calibrate_split': [False],
        'algorithm_args.calibration_indices': [[0, 2, 4]],
        'seedid': [2000, 2001, 2002],
        'layer_norm': [True],
        'freeze_decoder': [True],
        'freeze_rf': [True],
        'trainer_kwargs.use_supervised': ['calibrate_kl'],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)


    def process_args(variant):
        variant['env_config']['seedid'] = variant['seedid']
        if not args.use_ray:
            variant['render'] = args.no_render


    args.process_args = process_args

    run_exp(experiment, variants, args)
