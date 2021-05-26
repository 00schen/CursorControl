import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlpPolicy, VAE
from rl.misc.calibration_rl_algorithm import BatchRLAlgorithm as TorchCalibrationRLAlgorithm
from rl.misc.calibration_rl_algorithm_awr import BatchRLAlgorithm as TorchCalibrationRLAlgorithmAWR

from rl.policies import EncDecPolicy, CalibrationPolicy
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


def experiment(variant):
    import torch as th

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
    rf = loaded['trainer/rf']
    encoder = VAE(input_size=sum(env.feature_sizes.values()) + env.observation_space.low.size,
                  latent_size=variant['latent_size'],
                  encoder_hidden_sizes=[M],
                  decoder_hidden_sizes=[M]
                  ).to(ptu.device)
    qf = loaded['trainer/qf']
    target_qf = loaded['trainer/target_qf']
    if 'trainer/encoder' in loaded.keys():
        prev_encoder = loaded['trainer/encoder'].to(ptu.device)
    else:
        prev_encoder = None
    recon_decoder = ConcatMlpPolicy(input_size=variant['latent_size'],
                                    output_size=sum(env.feature_sizes.values()),
                                    hidden_sizes=[M],
                                    )

    optim_params = list(encoder.parameters())
    if not variant['freeze_decoder']:
        optim_params += list(qf.parameters())
    if not variant['freeze_rf']:
        optim_params += list(rf.parameters())
    optimizer = optim.Adam(
        optim_params,
        lr=variant['qf_lr'],
    )

    eval_policy = EncDecPolicy(
        qf,
        list(env.feature_sizes.keys()),
        encoder=encoder,
        sample=True,
        latent_size=variant['latent_size'],
    )
    eval_path_collector = FullPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False
    )
    expl_policy = EncDecPolicy(
        qf=qf,
        features_keys=list(env.feature_sizes.keys()),
        encoder=encoder,
        sample=True,
        latent_size=variant['latent_size'],
        **variant['expl_kwargs']
    )
    calibration_policy = CalibrationPolicy(
        qf=qf,
        features_keys=list(env.feature_sizes.keys()),
        env=env,
        eps=1,
        encoder=encoder,
        prev_encoder=prev_encoder,
        logit_scale=variant['expl_kwargs']['logit_scale']
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
    trainer = LatentEncDecCQLTrainer(
        rf=rf,
        encoder=encoder,
        recon_decoder=recon_decoder,
        prev_encoder=prev_encoder,
        qf=qf,
        target_qf=target_qf,
        optimizer=optimizer,
        latent_size=variant['latent_size'],
        **variant['trainer_kwargs']
    )
    calibration_buffer = ModdedReplayBuffer(
        variant['replay_buffer_size'],
        env,
        sample_base=0,
        latent_size=variant['latent_size']
    )

    alg_class = TorchCalibrationRLAlgorithmAWR if 'AWR' in variant['trainer_kwargs']['use_supervised'] else \
        TorchCalibrationRLAlgorithm
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
    parser.add_argument('--exp_name', default='calibrate')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    args, _ = parser.parse_known_args()
    main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200
    variant = dict(
        pretrain_path=f'{args.env_name}_params_s1_dense_encoder_5.pkl',
        latent_size=3,
        layer_size=64,
        expl_kwargs=dict(
            eps=0,
            logit_scale=-1
        ),
        replay_buffer_size=int(1e4 * path_length),
        trainer_kwargs=dict(
            target_update_period=1,
            qf_criterion=None,
            qf_lr=5e-4,
            discount=0.99,
            add_ood_term=-1,
            temp=10,
            min_q_weight=0,
            sample=True,
            beta=0.01,
            pos_weight=1
        ),
        algorithm_args=dict(
            batch_size=256,
            max_path_length=path_length,
            num_epochs=100,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=1,
            num_train_loops_per_epoch=1,
            collect_new_paths=True,
            num_trains_per_train_loop=1,
            # min_num_steps_before_training=1000
            trajs_per_index=5,
            calibration_indices=None,
            pretrain_steps=500,
            max_failures=10
        ),

        env_config=dict(
            env_name=args.env_name,
            step_limit=path_length,
            env_kwargs=dict(success_dist=.03, frame_skip=5, debug=False, num_targets=5, target_indices=[0, 2, 4]),

            action_type='disc_traj',
            smooth_alpha=.8,

            factories=[],
            adapts=['goal', 'static_gaze', 'reward'],
            gaze_dim=128,
            state_type=0,
            reward_max=0,
            reward_min=-5,
            reward_temp=2,
            reward_offset=-0.1,
            reward_type='sparse',
            gaze_path=f'{args.env_name}_gaze_data_train.h5',
            eval_gaze_path=f'{args.env_name}_gaze_data_eval.h5'
        )
    )
    variants = []

    search_space = {
        'seedid': [2000, 2001, 2002],
        'layer_norm': [True],
        'trainer_kwargs.soft_target_tau': [1e-2],
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
        variant['trainer_kwargs']['learning_rate'] = variant['trainer_kwargs'].pop('qf_lr')
        variant['qf_lr'] = variant['trainer_kwargs']['learning_rate']
        variant['env_config']['seedid'] = variant['seedid']
        if not args.use_ray:
            variant['render'] = args.no_render


    args.process_args = process_args

    run_exp(experiment, variants, args)
