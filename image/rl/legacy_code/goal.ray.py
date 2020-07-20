import gym
from gym import spaces
import assistive_gym
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf
import numpy as np
import pickle
from copy import deepcopy

import argparse

import ray

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray import tune
from ray.tune import registry
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str)
parser.add_argument('--local_dir', default="~/share/image/rl/test")
parser.add_argument('--exp_name', default="test")
parser.add_argument('--num_gpus', type=int, default=4)
args, _ = parser.parse_known_args()

from utils import *

def timer_factory(base):
	class Timer(base):
		def step(self,action):
			obs,r,done,info = super().step(action)

			self.step_count += 1
			if self.step_count >= 200:
				done = True
			return obs,r,done,info

		def reset(self):
			self.step_count = 0
			return super().reset()
	return Timer

class CustomCallback(DefaultCallbacks):
	def on_episode_step(self, worker, base_env, episode, **kwargs):
		env = base_env.get_unwrapped()[0]
		episode.custom_metrics["num_successes"] = env.task_success
		episode.custom_metrics["success_rate"] = env.task_success > 0
		if "distance_to_target" in episode.custom_metrics:
			episode.custom_metrics["distance_to_target"] = min(episode.custom_metrics["distance_to_target"],
														np.linalg.norm(env.target_pos - env.tool_pos))
		else:
			episode.custom_metrics["distance_to_target"] = np.linalg.norm(env.target_pos - env.tool_pos)
		# episode.custom_metrics["laptop_count"] = env.contact_laptop_count
		# episode.custom_metrics["laptop_move"] = env.laptop_move

if __name__ == "__main__":
	args = parser.parse_args()

	ray.init(temp_dir='/tmp/ray_exp',num_cpus=30, num_gpus=args.num_gpus)

	registry.register_env("Laptop", lambda kwargs: timer_factory(norm_factory(assistive_gym.LaptopJacoEnv))(**kwargs))
	registry.register_env("Reach", lambda kwargs: timer_factory(norm_factory(assistive_gym.ReachJacoEnv))(**kwargs))

	sched = AsyncHyperBandScheduler(
        time_attr="timesteps_total",
        metric="episode_reward_mean",
        mode="max",
        max_t=int(2.5e6),
        grace_period=int(5e5))

	space = {
        "optimization": hp.choice("optimization",[{
			"actor_learning_rate": hp.uniform("actor_learning_rate",1e-5,5e-4),
			"critic_learning_rate": hp.uniform("critic_learning_rate",1e-5,5e-4),
			"entropy_learning_rate": hp.uniform("entropy_learning_rate",1e-5,5e-4),
    	}]),
        "env_config": hp.choice("env_config",[{
			"reward_weights": hp.choice("reward_weights",[{
				"distance_weight": hp.uniform("distance_weight",1,10),
				"action_weight": hp.uniform("action_weight",0,1),
				# "target_weight": hp.uniform("target_weight",1,10),
				# "contact_weight": hp.uniform("contact_weight",1,10),
			}])
		}]),
    }
	# current_best_params = [
    # {
    #     "optimization": {
	# 		"actor_learning_rate": 1e-4,
	# 		"critic_learning_rate": 1e-4,
	# 		"entropy_learning_rate": 1e-4,
    # 	},
    #     "env_config": {
	# 		"reward_weights": {
	# 			"distance_weight": .9,
	# 			"action_weight": .01,
	# 			"target_weight": 9.9,
	# 			"contact_weight": 9.9,
	# 		}
	# 	},
    # }]
	algo = HyperOptSearch(
        space,
        metric="episode_reward_mean",
        mode="max",
        # points_to_evaluate=current_best_params
		)

	config = COMMON_CONFIG.copy()
	config.update(sac.DEFAULT_CONFIG)
	config.update({
		"env": args.env,
		# "env_config": {
		# 	"reward_weights": {
		# 		"distance_weight": 1,
		# 		"action_weight": .01,
		# 		"target_weight": 10,
		# 		"contact_weight": 10,
		# 	}
		# },

		# "optimization": {
		# 	"actor_learning_rate": 1e-4,
		# 	"critic_learning_rate": 1e-4,
		# 	"entropy_learning_rate": 1e-4,
    	# },

		"Q_model": {
			"fcnet_activation": "relu",
			"fcnet_hiddens": [512,512,512],
		},
		"policy_model": {
			"fcnet_activation": "relu",
			"fcnet_hiddens": [512,512,512],
		},
		"normalize_actions": True,
		"grad_clip": 1000,
		"train_batch_size": 256,

		"callbacks": CustomCallback,
		"eager": True,
		"num_gpus": 1,
	})
	config["tf_session_args"]["gpu_options"]["allow_growth"] = True

	stop = {
		"timesteps_total": int(5e6),
	}

	results = tune.run(CustomTrainable, name=args.exp_name, local_dir=args.local_dir,
													 num_samples=50,
													 checkpoint_freq=int(1e4),
													 config=config, stop=stop,
													 search_alg=algo,
													 scheduler=sched, 
													 verbose=1)

	ray.shutdown()