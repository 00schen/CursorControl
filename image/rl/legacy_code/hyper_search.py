import gym
import time
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf
from copy import deepcopy,copy

import assistive_gym
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

from utils import *
from envs import *

# from stable_baselines3.common import buffers
# buffers.ReplayBuffer = IndexReplayBuffer
# from stable_baselines3.awac import MlpPolicy, AWAC
from stable_baselines3.sac import MlpPolicy, SAC
from stable_baselines3.common.callbacks import BaseCallback,CallbackList
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir',default='~/share/image/rl/test',help='dir to save trials')
parser.add_argument('--config', type=int,)
parser.add_argument('--num_gpus', type=int, default=4, help='dir to save trials')
parser.add_argument('--gpu', type=float, )
args, _ = parser.parse_known_args()
dirname = os.path.abspath('')

def run(config, reporter):
	env_config = config['env_config']
	env_config.update({
					# 'threshold': config['threshold'],
					# 'blank': config['blank'],
					# 'input_penalty': config['input_penalty'],
					'action_penalty': config['action_penalty'],
					'num_obs': int(config['num_obs']),
					# 'num_nonnoop': int(config['num_nonnoop']),
					})

	logdir = tune.get_trial_dir()
	os.makedirs(logdir, exist_ok=True)
	env = make_vec_env(lambda: config['wrapper'](env_config['env_name'])(env_config),monitor_dir=logdir)
	env = VecNormalize(env,norm_reward=False)
	env.seed(config['seed'])
	model = SAC(MlpPolicy,env,train_freq=-1,n_episodes_rollout=1,gradient_steps=1,gamma=1.,
				verbose=1,tensorboard_log=logdir,seed=config['seed'],
				learning_rate=10**config['lr'],
				# learning_rate=10**config['lr'], beta=config['beta'], ent_coef=0,
				policy_kwargs={'net_arch': [config['layer_size']]*config['layer_depth']})
	
	# awac_path_loader(env,np.load(os.path.join(config['demo_file_path'],config['bc_file']),allow_pickle=True),model.bc_buffer)
	# awac_path_loader(env,np.load(os.path.join(config['demo_file_path'],config['offpolicy_file']),allow_pickle=True),model.replay_buffer)

	class ReportCallback(BaseCallback):
		def __init__(self, verbose=0):
			super().__init__(verbose)
		def _on_step(self):
			env = self.training_env.envs[0]
			if self.n_calls % 1000 == 0:
				tune.report(
					# t=env.scheduler.t,
					success_rate=np.mean(env.success_count),
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
	time_steps = int(2e6)
	model.learn(total_timesteps=time_steps,callback=callback,)
	
if __name__ == "__main__":
	args = parser.parse_args()
	ray.init(temp_dir='/tmp/ray_exp', num_gpus=args.num_gpus)

	sched = ASHAScheduler(
		time_attr="timesteps_total",
		metric="success_rate",
		# metric="t",
		mode="max",
		max_t=int(1e6),
		grace_period=int(4e5))

	space = {
		"lr": hp.uniform("lr",-3,-2.3),
		# "input_penalty": hp.uniform("input_penalty",1,50),
		# "threshold": hp.uniform("threshold", -.6,.6),
		# "blank": hp.uniform("blank", 0,1),
		"num_obs": hp.randint("num_obs",5,21),
		"action_penalty": hp.uniform("action_penalty",0,5),
		# "beta": hp.uniform("beta",0,100),
		# "num_nonnoop": hp.randint("num_nonnoop",5,21),
		# "layer_size": hp.choice("layer_size",[256,512]),
		# "layer_depth": hp.choice("layer_depth",[3,4,5]),
	}
	algo = HyperOptSearch(
		space,
		metric="success_rate",
		# metric="t",
		mode="max",
		# points_to_evaluate=current_best_params
		)

	stop = {
		"timesteps_total": int(2e6),
	}

	env_config = deepcopy(default_config)
	s_config = deepcopy(env_config)
	f_config = deepcopy(env_config)
	l_config = deepcopy(env_config)
	r_config = deepcopy(env_config)
	ls_config = deepcopy(env_config)
	s_config.update(env_map['ScratchItch'])
	f_config.update(env_map['Feeding'])
	l_config.update(env_map['Laptop'])
	r_config.update(env_map['Reach'])
	ls_config.update(env_map['LightSwitch'])

	"""trial configs"""
	trial = [
		{'exp_name': '7_20_0a', 'seed': 1000, 'env_config':{**ls_config,**{'oracle':'ded_target',},},
			'bc_file': "LightSwitch_demo1.npy", 'offpolicy_file': "LightSwitch_offpolicy1.npy"},
		{'exp_name': '7_20_0b', 'seed': 1000, 'env_config':{**ls_config,**{'oracle':'dd_target',},},
			'bc_file': "LightSwitch_demo1.npy", 'offpolicy_file': "LightSwitch_offpolicy1.npy"},
		{'exp_name': '7_20_0c', 'seed': 1000, 'env_config':{**ls_config,**{'oracle':'dd_target',},},
			'bc_file': "LightSwitch_demo2.npy", 'offpolicy_file': "LightSwitch_offpolicy2.npy"},

		{'exp_name': '7_20_1a', 'seed': 1000, 'env_config':{**l_config,**{'oracle':'ded_target',},},
			'bc_file': "Laptop_demo1.npy", 'offpolicy_file': "Laptop_offpolicy1.npy"},
		{'exp_name': '7_20_1b', 'seed': 1000, 'env_config':{**l_config,**{'oracle':'dd_target',},},
			'bc_file': "Laptop_demo1.npy", 'offpolicy_file': "Laptop_offpolicy1.npy"},
		{'exp_name': '7_20_1c', 'seed': 1000, 'env_config':{**l_config,**{'oracle':'dd_target',},},
			'bc_file': "Laptop_demo2.npy", 'offpolicy_file': "Laptop_offpolicy2.npy"},

		{'exp_name': '7_7_3a', 'seed': 1000, 'env_config':{**r_config,**{'oracle':'ded_target','env_kwargs': {'num_targets': 2,},},},
			'bc_file': "Reach_demo1.npy", 'offpolicy_file': "Reach_offpolicy1.npy"},
		{'exp_name': '7_7_3b', 'seed': 1000, 'env_config':{**r_config,**{'oracle':'dd_target','env_kwargs': {'num_targets': 2,},},},
			'bc_file': "Reach_demo1.npy", 'offpolicy_file': "Reach_offpolicy1.npy"},
		{'exp_name': '7_7_3c', 'seed': 1000, 'env_config':{**r_config,**{'oracle':'dd_target','env_kwargs': {'num_targets': 2,},},},
			'bc_file': "Reach_demo2.npy", 'offpolicy_file': "Reach_offpolicy2.npy"},
	][args.config]

	trial['env_config'].update({
		'oracle_size': 6,
		'num_nonnoop': 10,
		"input_penalty": 2,
		'action_type': 'trajectory',
	})
	trial.update({'curriculum': False, 'wrapper': default_class,
				'layer_size': 256, 'layer_depth': 3,'demo_file_path': os.path.join(os.path.abspath(''),'demos')})

	results = tune.run(run, name= trial['exp_name'], local_dir=args.local_dir,
													 num_samples=100,
													 config=trial, stop=stop,
													 search_alg=algo,
													 scheduler=sched, 
													 resources_per_trial={'cpu':1,'gpu':args.gpu},
													 verbose=1)

	ray.shutdown()