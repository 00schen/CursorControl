import rlkit.torch.pytorch_util as ptu
from rlkit.pythonplusplus import merge_recursive_dicts
from rlkit.torch.networks import VAE
from rl.misc.calibration_rl_algorithm import BatchRLAlgorithm as TorchCalibrationRLAlgorithm

from rl.policies import CalibrationSACPolicy, EncDecSACPolicy
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


def experiment(variant):
    import torch as th

    expl_config = deepcopy(variant['env_config'])
    if 'calibrate' in variant['trainer_kwargs']['use_supervised']:
        expl_config['factories'] += ['session']
    env = default_overhead(expl_config)
    env.seed(variant['seedid'])
    eval_config = deepcopy(variant['env_config'])
    eval_config = merge_recursive_dicts(eval_config, variant['eval_config'])
    eval_config['gaze_path'] = eval_config['eval_gaze_path']
    eval_env = default_overhead(eval_config)
    eval_env.seed(variant['seedid'] + 1)

    M = variant["layer_size"]

    file_name = os.path.join('util_models', variant['pretrain_path'])
    loaded = th.load(file_name)
    vae = VAE(input_size=sum(env.feature_sizes.values()) + env.observation_space.low.size,
              latent_size=variant['latent_size'],
              encoder_hidden_sizes=[M] * variant['n_layers'],
              decoder_hidden_sizes=[M] * variant['n_layers']
              ).to(ptu.device)
    policy = loaded['trainer/policy']
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

    expl_policy = EncDecSACPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        vae=vae,
        incl_state=True,
        sample=variant['trainer_kwargs']['sample'],
        deterministic=True,
        latent_size=variant['latent_size'],
    )

    eval_policy = EncDecSACPolicy(
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

    calibration_policy = CalibrationSACPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        env=env,
        vae=vae,
        prev_vae=prev_vae,
        incl_state=True
    )
    expl_path_collector = FullPathCollector(
        env,
        expl_policy,
        save_env_in_snapshot=False,
    )
    calibration_path_collector = FullPathCollector(
        env,
        calibration_policy,
        save_env_in_snapshot=False,
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
        optimizer=optimizer,
        latent_size=variant['latent_size'],
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

    alg_class = TorchCalibrationRLAlgorithm
    algorithm = alg_class(
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
    parser.add_argument('--exp_name', default='calibrate_sac_5')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    args, _ = parser.parse_known_args()
    main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200
    variant = dict(
        pretrain_path=f'{args.env_name}_params_s1_1e-1_sac.pkl',
        latent_size=3,
        layer_size=64,
        replay_buffer_size=int(1e4 * path_length),
        keep_calibration_data=True,
        trainer_kwargs=dict(
            beta=0.01,
            grad_norm_clip=1
        ),
        algorithm_args=dict(
            batch_size=256,
            # max_path_length=path_length,
            num_epochs=250,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1,
            num_train_loops_per_epoch=1,
            collect_new_paths=True,
            num_trains_per_train_loop=1,
            pretrain_steps=1000,
            # max_failures=1,
        ),

        env_config=dict(
            env_name=args.env_name,
            terminate_on_failure=True,
            step_limit=path_length,
            env_kwargs=dict(success_dist=.03, frame_skip=5, debug=False,),

            action_type='joint',
            smooth_alpha=1,

            factories=[],
            adapts=['goal', 'static_gaze', 'reward'],
            gaze_dim=128,
            state_type=0,
            reward_max=0,
            reward_min=-1,
            reward_temp=1,
            reward_offset=-0.1,
            reward_type='sparse',
            gaze_path=f'{args.env_name}_gaze_data_train.h5',
            eval_gaze_path=f'{args.env_name}_gaze_data_eval.h5'
        ),
        eval_config=dict(
            env_kwargs=dict(),
        )
    )
    variants = []

    search_space = {
        'seedid': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,],
        'n_layers': [1],
        'algorithm_args.trajs_per_index': [3],
        'lr': [5e-4],
        'trainer_kwargs.sample': [True],
        'algorithm_args.calibrate_split': [False,],
        'algorithm_args.calibration_indices': [[0,3],[0,1,2,3]],
        'algorithm_args.max_path_length': [path_length,],
        'algorithm_args.max_failures': [5],
        'eval_config.env_kwargs.target_indices': [[1,2],[0,3]],
        'freeze_decoder': [True],
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
