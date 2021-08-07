
from rl.scripts.run_util import run_exp
from rl.scripts.sac_s2_calibrate_base import experiment as base_experiment
from rl.scripts.sac_s2_calibrate_base_np import experiment as np_experiment

from pathlib import Path
import rlkit.util.hyperparameter as hyp
import argparse
import numpy as np

def experiment(variant):
    if variant['use_np']:
        np_experiment(variant)
    else:
        base_experiment(variant)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='PointReach')
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
                            curriculum=args.curriculum,),
                            #  always_reset=False),
            action_type='joint',
            smooth_alpha=1,
            factories=[],
            adapts=['goal', 'sim_target'],
            feature='sub_goal',
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
        'seedid': [0, 1, 2,],
        # 'env_config.adapts': [['goal', 'sim_target'], ['goal', 'sim_keyboard']],
        'env_config.env_kwargs.always_reset': [True, False],
        'use_np': [True, False]
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
