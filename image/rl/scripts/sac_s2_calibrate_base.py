import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import VAE
from rl.misc.calibration_rl_algorithm import BatchRLAlgorithm as TorchCalibrationRLAlgorithm

from rl.policies import CalibrationPolicy, EncDecPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.trainers import LatentEncDecSACTrainer
from rl.replay_buffers import ModdedReplayBuffer

import os
from pathlib import Path
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
        random_latent=variant.get('random_latent', False)
    )

    eval_policy = EncDecPolicy(
        policy=policy,
        features_keys=list(env.feature_sizes.keys()),
        vaes=vaes,
        incl_state=variant['incl_state'],
        sample=variant['sample'],
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
        obs_feature_sizes={'sub_goal':3}
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

    if variant['keep_calibration_data']:
        calibration_buffer = replay_buffer
    else:
        calibration_buffer = ModdedReplayBuffer(
            variant['replay_buffer_size'],
            env,
            sample_base=0,
            latent_size=variant['latent_size'],
            store_latents=True,
            obs_feature_sizes={'sub_goal':3}
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
