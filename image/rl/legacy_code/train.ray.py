import gym
from gym import spaces
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf
import assistive_gym
import numpy as np
import pickle
from copy import deepcopy

import argparse

import ray
from ray import tune
from ray.rllib.agents import sac
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import registry
from baselines.common.running_mean_std import RunningMeanStd
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str)
parser.add_argument("--stop-timesteps", type=int, default=int(1e6))
parser.add_argument('--local_dir', default="~/share/image/rl/test")
parser.add_argument('--exp_name', default="test")
args, _ = parser.parse_known_args()

class RewardRolloutCallback(DefaultCallbacks):
	def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
		env = base_env.get_unwrapped()[0]

		episode.custom_metrics["num_successes"] = env.env.env.task_success
		episode.custom_metrics["success_rate"] = env.env.env.task_success > 0
		episode.custom_metrics["t"] = env.t
	def on_sample_end(self, worker, samples, **kwargs):
		if samples["infos"][-1]["task_success"] > 0:
			samples["rewards"][:-1] += 100*np.power(np.arange(len(samples["rewards"]),1,-1),.995)

if __name__ == "__main__":
	ray.init(temp_dir='/tmp/ray_exp',num_gpus=4)

	registry.register_env("MovingEnd", lambda config: norm_factory(MovingEndCurriculum)(config))
	registry.register_env("MovingInit", lambda config: norm_factory(MovingInitCurriculum)(config))

	curriculum_default_config = {'num_obs': 5, 'oracle': 'trajectory','coop': False, 'action_type': 'target'}
	env_config = deepcopy(default_config)
	env_config.update(curriculum_default_config)
	scratch_itch_config = deepcopy(env_config)
	scratch_itch_config.update(env_map["ScratchItch"])
	feeding_config = deepcopy(env_config)
	feeding_config.update(env_map["Feeding"])

	config = {
		"env": "MovingEnd",
		"env_config": tune.grid_search([feeding_config,scratch_itch_config]),

		"Q_model": {
			"fcnet_activation": "relu",
			"fcnet_hiddens": [512,512,512],
		},
		"policy_model": {
			"fcnet_activation": "relu",
			"fcnet_hiddens": [512,512,512],
		},
		"optimization": {
			"actor_learning_rate": 1e-4,
			"critic_learning_rate": 1e-4,
			"entropy_learning_rate": 1e-4,
    	},
		"normalize_actions": True,
		"grad_clip": 1000,
		"rollout_fragment_length": env_config["step_limit"],
		"batch_mode": "complete_episodes",
		"train_batch_size": 512,

		"num_gpus": 1,
		"num_workers": 1,
		"callbacks": tune.grid_search([RewardRolloutCallback,CustomMetricCallback]),
		"eager": True,
	}

	stop = {
		"timesteps_total": args.stop_timesteps,
	}

	results = tune.run(CustomTrainable, name=args.exp_name, local_dir=args.local_dir,
													 checkpoint_freq=int(100),
													 config=config,
													 stop=stop, 
													 verbose=1)

	ray.shutdown()