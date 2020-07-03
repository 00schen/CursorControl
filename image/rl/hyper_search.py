import gym
import time
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf
from copy import deepcopy

from stable_baselines3.sac import MlpPolicy
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import BaseCallback,CallbackList
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import assistive_gym
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

from utils import *
from envs import *

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir',default='~/share/image/rl/test',help='dir to save trials')
parser.add_argument('--config', type=int,)
parser.add_argument('--num_gpus', type=int, default=4, help='dir to save trials')
parser.add_argument('--gpu', type=float, )
args, _ = parser.parse_known_args()
dirname = os.path.abspath('')

def run(config, reporter):
	env_config = config['env_config']
	logdir = tune.get_trial_dir()
	os.makedirs(logdir, exist_ok=True)
	env = make_vec_env(lambda: config['wrapper'](env_config),monitor_dir=logdir)
	env = VecNormalize(env,norm_reward=False)
	env.seed(config['seed'])
	model = SAC(MlpPolicy,env,train_freq=-1,n_episodes_rollout=1,gradient_steps=100,gamma=1.,
				verbose=1,tensorboard_log=logdir,seed=config['seed'],
				learning_rate=config['lr'],
				policy_kwargs={'net_arch': [config['layer_size']]*config['layer_depth']})
				

	class ReportCallback(BaseCallback):
		def __init__(self, verbose=0):
			super().__init__(verbose)
		def _on_step(self):
			env = self.training_env.envs[0]
			if self.n_calls % 1000 == 0:
				tune.report(t=env.t,
						timesteps_total=self.num_timesteps)
			return True
	class TuneCheckpointCallback(BaseCallback):
		def __init__(self, save_freq, verbose=0):
			super().__init__(verbose)
			self.save_freq = save_freq
		def _on_step(self):
			if self.n_calls % self.save_freq == 0:
				path = tune.get_trial_dir()
				self.training_env.save(os.path.join(path,f"norm.{self.num_timesteps}"))
				self.model.save(os.path.join(path,f"model.{self.num_timesteps}"))
				# model.save_replay_buffer(os.path.join(path,f"replay.{self.num_timesteps}"))
			return True			
	callback = CallbackList([
		TuneCheckpointCallback(save_freq=int(5e4)),
		TensorboardCallback(curriculum=config['curriculum']),
		ReportCallback(),
	])
	time_steps = int(1.2e6)
	model.learn(total_timesteps=time_steps,callback=callback,)
	
if __name__ == "__main__":
	args = parser.parse_args()
	ray.init(temp_dir='/tmp/ray_exp', num_gpus=args.num_gpus)

	sched = ASHAScheduler(
		time_attr="timesteps_total",
		metric="t",
		mode="max",
		max_t=int(1.2e6),
		grace_period=int(4e5))

	space = {
		"lr": hp.uniform("lr",1e-3,1e-2),
		# "layer_size": hp.choice("layer_size",[256,512]),
		"layer_depth": hp.choice("layer_depth",[2,3,4,5]),
	}
	current_best_params = [{
		"lr": 1e-4,
		"layer_size": 256,
		"layer_depth": 3,
	}]
	algo = HyperOptSearch(
		space,
		metric="t",
		mode="max",
		# points_to_evaluate=current_best_params
		)

	stop = {
		"timesteps_total": int(2e6),
	}


	env_config = deepcopy(default_config)
	r_config = deepcopy(env_config)
	r_config.update(env_map['Reach'])

	"""trial configs"""
	trial = [
		{'exp_name': '7_2_0', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'dist_hot_cold',}}},
		{'exp_name': '7_2_1', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'rad_hot_cold',}}},
		{'exp_name': '7_2_2', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'dist_discrete_traj',}}},
		{'exp_name': '7_2_3', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'rad_discrete_traj',}}},
	][args.config]

	trial['env_config'].update({
		'end_early': True,
		'noise': False,
		'oracle_size': 10,
		'phi':lambda d:0,
		'env_kwargs': {'num_targets':2,},
	})
	trial['env_config'].update(action_map['trajectory'])
	trial.update({'curriculum': True, 'wrapper': MovingInit, 'layer_size': 256})

	results = tune.run(run, name= trial['exp_name'], local_dir=args.local_dir,
													 num_samples=50,
													 config=trial, stop=stop,
													 search_alg=algo,
													 scheduler=sched, 
													 resources_per_trial={'cpu':1,'gpu':args.gpu},
													 verbose=1)

	ray.shutdown()