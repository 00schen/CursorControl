import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlpPolicy, VAE
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

from rl.policies import EncDecPolicy, KeyboardPolicy
from rl.path_collectors import FullPathCollector
from rl.misc.env_wrapper import default_overhead
from rl.trainers import EncDecCQLTrainer
from rl.replay_buffers import HERReplayBuffer, ModdedReplayBuffer
from rl.scripts.run_util import run_exp

import os
from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
from torch import optim


def experiment(variant):
    import torch as th

    env = default_overhead(variant['env_config'])
    env.seed(variant['seedid'])
    eval_config = variant['env_config'].copy()
    eval_env = default_overhead(eval_config)
    eval_env.seed(variant['seedid'] + 1)

    obs_dim = env.observation_space.low.size + variant['latent_size']
    action_dim = env.action_space.low.size
    M = variant["layer_size"]

    if not variant['from_pretrain']:
        rf = ConcatMlpPolicy(input_size=obs_dim,
                             output_size=1,
                             hidden_sizes=[M, M],
                             layer_norm=variant['layer_norm'],
                             )
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
        encoder = VAE(input_size=sum(env.feature_sizes.values()),
                      latent_size=variant['latent_size'],
                      encoder_hidden_sizes=[64],
                      decoder_hidden_sizes=[64]
                      ).to(ptu.device)
    else:
        file_name = os.path.join('image/util_models', variant['pretrain_path'])
        loaded = th.load(file_name)
        rf = ConcatMlpPolicy(input_size=obs_dim,
                             output_size=1,
                             hidden_sizes=[M, M],
                             layer_norm=variant['layer_norm'],
                             )
        qf = loaded['trainer/qf']
        target_qf = loaded['trainer/target_qf']
        encoder = loaded['trainer/encoder']
    optimizer = optim.Adam(
        # list(rf.parameters()) + list(qf.parameters()),
        list(rf.parameters()) + list(qf.parameters()) + list(encoder.parameters()),
        lr=variant['qf_lr'],
    )

    eval_policy = EncDecPolicy(
        qf,
        list(env.feature_sizes.keys()),
        encoder=encoder,
        incl_state=False,
        sample=False,
    )
    eval_path_collector = FullPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False
    )
    expl_policy = EncDecPolicy(
        qf,
        list(env.feature_sizes.keys()),
        encoder=encoder,
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
    trainer = EncDecCQLTrainer(
        rf=rf,
        qf=qf,
        target_qf=target_qf,
        encoder=encoder,
        optimizer=optimizer,
        latent_size=variant['latent_size'],
        **variant['trainer_kwargs']
    )
    replay_buffer = variant['buffer_type'](
        variant['replay_buffer_size'],
        env,
        sample_base=0,
        #k=variant['her_k']
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
    # path_loader = SimplePathLoader(
    #     demo_path=variant['demo_paths'],
    #     demo_path_proportion=variant['demo_path_proportions'],
    #     replay_buffer=replay_buffer,
    # )
    # path_loader.load_demos()

    if variant.get('render', False):
        env.render('human')
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', )
    parser.add_argument('--exp_name', default='pretrain')
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
        layer_size=128,
        # her_k=4,
        expl_kwargs=dict(
        ),
        trainer_kwargs=dict(
            target_update_period=1,
            qf_criterion=None,
            qf_lr=5e-4,
            discount=0.99,
            add_ood_term=-1,
            temp=1,
            min_q_weight=0,
            sample=True,
            train_encoder_on_rf=False
        ),
        algorithm_args=dict(
            batch_size=256,
            max_path_length=path_length,
            num_epochs=200,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=500,
            num_train_loops_per_epoch=10,
            collect_new_paths=True,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=10000
        ),
        # algorithm_args=dict(
        #     batch_size=256,
        #     max_path_length=path_length,
        #     num_epochs=10,
        #     num_eval_steps_per_epoch=5000,
        #     num_expl_steps_per_train_loop=0,
        #     num_train_loops_per_epoch=1,
        #     collect_new_paths=False,
        #     num_trains_per_train_loop=0,
        #     min_num_steps_before_training=1
        # ),

        # demo_paths=[
        #     os.path.join(main_dir, "demos", f"{args.env_name}_model_on_policy_5000_debug.npy"),
        # ],

        env_config=dict(
            env_name=args.env_name,
            step_limit=path_length,
            env_kwargs=dict(success_dist=.03, frame_skip=5, debug=False, num_targets=3),
            action_type='disc_traj',
            smooth_alpha=.8,
            factories=[],
            adapts=['goal', 'reward'],
            gaze_dim=128,
            state_type=0,
            reward_max=0,
            reward_min=-3,
            reward_type='custom_switch',
            reward_temp=2,
            reward_offset=-0.1
        )
    )
    search_space = {
        'seedid': [2000],
        'from_pretrain': [False],
        'layer_norm': [True],
        'expl_kwargs.logit_scale': [10],
        'expl_kwargs.eps': [0.1],
        'trainer_kwargs.soft_target_tau': [1e-2],
        # 'demo_path_proportions': [[int(5e3)], ],
        'trainer_kwargs.beta': [.01],
        'buffer_type': [ModdedReplayBuffer],
        'replay_buffer_size': [200000],
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
        variant['env_config']['seedid'] = variant['seedid']
        if not args.use_ray:
            variant['render'] = args.no_render


    args.process_args = process_args

    run_exp(experiment, variants, args)
