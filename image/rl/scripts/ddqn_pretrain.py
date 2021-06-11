import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlpPolicy, VAE
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import EncDecQfPolicy, KeyboardPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.misc.simple_path_loader import SimplePathLoader
from rl.trainers import EncDecDDQNTrainer
from rl.replay_buffers import ModdedReplayBuffer
from rl.scripts.run_util import run_exp

import os
from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
from torch import optim
from functools import reduce
import operator


def experiment(variant):
    import torch as th

    env = default_overhead(variant['env_config'])
    env.seed(variant['seedid'])
    eval_config = variant['env_config'].copy()
    eval_env = default_overhead(eval_config)
    eval_env.seed(variant['seedid'] + 1)

    obs_dim = env.observation_space.low.size + reduce(operator.mul,
                                                      getattr(env.base_env, 'goal_set_shape', (0,)),
                                                      1) + variant['latent_size']
    action_dim = env.action_space.low.size
    M = variant["layer_size"]

    if not variant['from_pretrain']:
        qf = ConcatMlpPolicy(input_size=obs_dim,
                             output_size=action_dim,
                             hidden_sizes=[M, M, M, M],
                             layer_norm=variant['layer_norm'],
                             )
        target_qf = ConcatMlpPolicy(input_size=obs_dim,
                                    output_size=action_dim,
                                    hidden_sizes=[M, M, M, M],
                                    layer_norm=variant['layer_norm'],
                                    )
        vae = VAE(input_size=sum(env.feature_sizes.values()),
                  latent_size=variant['latent_size'],
                  encoder_hidden_sizes=[64],
                  decoder_hidden_sizes=[64]
                  ).to(ptu.device)
    else:
        file_name = os.path.join('image/util_models', variant['pretrain_path'])
        loaded = th.load(file_name)
        qf = loaded['trainer/qf']
        target_qf = loaded['trainer/target_qf']
        vae = loaded['trainer/vae']
    optimizer = optim.Adam(
        list(list(qf.parameters()) + list(vae.encoder.parameters())),
        lr=variant['lr'],
    )

    eval_policy = EncDecQfPolicy(
        qf,
        list(env.feature_sizes.keys()),
        vae=vae,
        incl_state=False,
        sample=False,
    )
    eval_path_collector = FullPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False
    )
    expl_policy = EncDecQfPolicy(
        qf,
        list(env.feature_sizes.keys()),
        vae=vae,
        logit_scale=variant['expl_kwargs']['logit_scale'],
        eps=variant['expl_kwargs']['eps'],
        incl_state=False,
        sample=False,
    )
    expl_path_collector = FullPathCollector(
        env,
        expl_policy,
        save_env_in_snapshot=False,
    )
    trainer = EncDecDDQNTrainer(
        qf=qf,
        target_qf=target_qf,
        vae=vae,
        optimizer=optimizer,
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
    parser.add_argument('--exp_name', default='pretrain_dqn')
    parser.add_argument('--no_render', action='store_false')
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--per_gpu', default=1, type=int)
    args, _ = parser.parse_known_args()
    main_dir = args.main_dir = str(Path(__file__).resolve().parents[2])

    path_length = 200
    variant = dict(
        pretrain_path=f'{args.env_name}_params_s1_dqn.pkl',
        latent_size=3,
        layer_size=128,
        lr=5e-4,
        expl_kwargs=dict(
        ),
        trainer_kwargs=dict(
            target_update_period=1,
            qf_criterion=None,
            discount=0.99,
            add_ood_term=-1,
            temp=1,
            sample=True,
        ),
        algorithm_args=dict(
            batch_size=256,
            max_path_length=path_length,
            num_epochs=1500,
            eval_paths=False,
            num_eval_steps_per_epoch=0,
            num_expl_steps_per_train_loop=1000,
            collect_new_paths=True,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=1000
        ),
        
        demo_paths=[
            os.path.join(main_dir, "demos", f"{args.env_name}_model_on_policy_5000_debug1.npy"),
        ], # no latent

        env_config=dict(
            terminate_on_failure=False,
            env_name=args.env_name,
            step_limit=path_length,
            goal_noise_std=0,
            env_kwargs=dict(frame_skip=5, debug=False, num_targets=5),
            action_type='disc_traj',
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
        'layer_norm': [True],
        'expl_kwargs.logit_scale': [10],
        'expl_kwargs.eps': [0.1],
        'trainer_kwargs.soft_target_tau': [1e-2],
        'demo_path_proportions': [[int(5e3)], ],
        'trainer_kwargs.beta': [.01],
        'buffer_type': [ModdedReplayBuffer],
        'replay_buffer_size': [500000],
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
