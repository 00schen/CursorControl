import gym
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf
import assistive_gym
import numpy as np

import argparse

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune import registry
from baselines.common.running_mean_std import RunningMeanStd

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--env", type=str)
parser.add_argument("--stop-timesteps", type=int, default=int(10e6))
parser.add_argument('--local_dir')
parser.add_argument('--exp_name')
args, _ = parser.parse_known_args()

env_map = {
		'LaptopJaco-v0': 21
	}

if __name__ == "__main__":
	args = parser.parse_args()

	ray.init(temp_dir='/tmp/ray_exp')

	registry.register_env("LaptopJaco-v0", lambda c: assistive_gym.LaptopJacoEnv(c['time_limit']))

	@ray.remote
	class RMSWrapper(RunningMeanStd):
		def update(self,arr):
			super().update(arr)
			return self.mean,self.var
	obs_rms = RMSWrapper.options(name="obs_rms").remote(shape=(env_map[args.env],))

	class CustomCallback(DefaultCallbacks):
		def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
			episode.custom_metrics['success_rate'] = base_env.get_unwrapped()[0].task_success > 0

		def on_sample_end(self, worker, samples, **kwargs):
			mean,var = ray.get(obs_rms.update.remote(samples['obs']))
			samples['obs'] = self._obfilt(samples['obs'],mean,var)
			samples['new_obs'] = self._obfilt(samples['new_obs'],mean,var)

		def _obfilt(self, obs, mean, var):
			obs = np.clip((obs - mean) / np.sqrt(var + 1e-8), -10, 10)
			return obs

	config = {
		"env": args.env,
		"env_config": {
			"time_limit": 200,
		},

		"gamma": 0.995,
		"kl_coeff": 1.0,
		"num_sgd_iter": 20,
		"lr": .0001,
		"num_workers": 16,
		"num_gpus": 2,
		"callbacks": CustomCallback,
		"batch_mode": "complete_episodes",
	}

	stop = {
		"timesteps_total": args.stop_timesteps,
	}

	results = tune.run(args.run, name=args.exp_name, local_dir=args.local_dir, config=config, stop=stop, verbose=1)

	ray.shutdown()