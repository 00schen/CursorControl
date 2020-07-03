import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
from copy import deepcopy

from stable_baselines3.sac import MlpPolicy
from stable_baselines3.sac import SAC
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from utils import *
from envs import *

import ray
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir', default="~/share/image/rl/test",help='dir to save trials')
parser.add_argument('--tag')
args, _ = parser.parse_known_args()

@ray.remote(num_cpus=1,num_gpus=.25)
class Runner:
	def run(self,config):
		env_config = config['env_config']
		logdir = os.path.join(args.local_dir,config['exp_name'])
		os.makedirs(logdir, exist_ok=True)
		env = config['wrapper']
		env = make_vec_env(lambda: env(env_config),monitor_dir=logdir)
		env = VecNormalize(env,norm_reward=False)
		env.seed(config['seed'])
		model = SAC(MlpPolicy, env, learning_rate=config['lr'], train_freq=-1, n_episodes_rollout=1, gradient_steps=200,
					verbose=1,tensorboard_log=logdir)

		# path = os.path.join(os.path.abspath(''),"replays",f"replay.{env_config['env_name'][0]}.traj")
		# with open(path, "rb") as file_handler:
		# 	buffer = pickle.load(file_handler)
		# buffer.__class__ = PostProcessingReplayBuffer
		# buffer.device = 'cuda'
		# if config['prime_buffer']:
		# 	model.replay_buffer = buffer
		# else:
		# 	model.replay_buffer.__class__ = PostProcessingReplayBuffer

		callback = CallbackList([
				# RewardRolloutCallback(relabel_all=config['relabel_all']),
				TensorboardCallback(curriculum=config['curriculum']),
				NormCheckpointCallback(save_freq=int(5e4), save_path=logdir),
				CheckpointCallback(save_freq=int(5e4), save_path=logdir),
			])
		time_steps = int(2e6)
		model.learn(total_timesteps=time_steps,callback=callback,tb_log_name=config['exp_name'])

if __name__ == "__main__":
	ray.init(temp_dir='/tmp/ray_exp1')
	curriculum_default_config = {}
	env_config = deepcopy(default_config)
	env_config.update(curriculum_default_config)

	s_config = deepcopy(env_config)
	f_config = deepcopy(env_config)
	l_config = deepcopy(env_config)
	r_config = deepcopy(env_config)
	ls_config = deepcopy(env_config)
	s_config.update(env_map['ScratchItch'])
	f_config.update(env_map['Feeding'])
	l_config.update(env_map['Laptop'])
	ls_config.update(env_map['LightSwitch'])
	r_config.update(env_map['Reach'])

	# """no reward term noised"""
	# trial_configs = [
	# 	{'exp_name': '7_1_18', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':2,'target_first':True},}}},
	# 	{'exp_name': '7_1_19', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':5,'target_first':True},}}},
	# 	{'exp_name': '7_1_20', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':10,'target_first':True},}}},
	# 	{'exp_name': '7_1_21', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':-1,'target_first':True},}}},

	# 	{'exp_name': '7_1_22', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':2,'target_first':True},}}},
	# 	{'exp_name': '7_1_23', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':5,'target_first':True},}}},
	# 	{'exp_name': '7_1_24', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':10,'target_first':True},}}},
	# 	{'exp_name': '7_1_25', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':-1,'target_first':True},}}},
	# ]
	# for trial in trial_configs:
	# 	trial['env_config'].update({'reward_type':'distance_target','phi':lambda d:0,})
	# 	trial['env_config'].update(action_map['trajectory'])
	# 	trial.update({'curriculum': True, 'lr': 1e-4})

	# """no reward term noised"""
	# trial_configs = [
	# 	{'exp_name': '7_1_18a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':2,'target_first':True},}}},
	# 	{'exp_name': '7_1_19a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':5,'target_first':True},}}},
	# 	{'exp_name': '7_1_20a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':10,'target_first':True},}}},
	# 	{'exp_name': '7_1_21a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':-1,'target_first':True},}}},

	# 	{'exp_name': '7_1_22a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':2,'target_first':True},}}},
	# 	{'exp_name': '7_1_23a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':5,'target_first':True},}}},
	# 	{'exp_name': '7_1_24a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':10,'target_first':True},}}},
	# 	{'exp_name': '7_1_25a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':-1,'target_first':True},}}},
	# ]
	# for trial in trial_configs:
	# 	trial['env_config'].update({'reward_type':'distance_target','phi':lambda d:0,})
	# 	trial['env_config'].update(action_map['trajectory'])
	# 	trial.update({'curriculum': True, 'lr': 1e-4})

	# """no reward term no noised"""
	# trial_configs = [
	# 	{'exp_name': '7_1_26', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':2,'target_first':True},}}},
	# 	{'exp_name': '7_1_27', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':5,'target_first':True},}}},
	# 	{'exp_name': '7_1_28', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':10,'target_first':True},}}},
	# 	{'exp_name': '7_1_29', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':-1,'target_first':True},}}},

	# 	{'exp_name': '7_1_30', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':2,'target_first':True},}}},
	# 	{'exp_name': '7_1_31', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':5,'target_first':True},}}},
	# 	{'exp_name': '7_1_32', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':10,'target_first':True},}}},
	# 	{'exp_name': '7_1_33', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':-1,'target_first':True},}}},
	# ]
	# for trial in trial_configs:
	# 	trial['env_config'].update({'reward_type':'distance_target','phi':lambda d:0,})
	# 	trial['env_config'].update(action_map['trajectory'])
	# 	trial.update({'curriculum': True, 'lr': 1e-4})

	# """no reward term noised"""
	# trial_configs = [
	# 	{'exp_name': '7_1_26a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':2,'target_first':True},}}},
	# 	{'exp_name': '7_1_27a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':5,'target_first':True},}}},
	# 	{'exp_name': '7_1_28a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':10,'target_first':True},}}},
	# 	{'exp_name': '7_1_29a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'target','env_kwargs': {'num_targets':-1,'target_first':True},}}},

	# 	{'exp_name': '7_1_30a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':2,'target_first':True},}}},
	# 	{'exp_name': '7_1_31a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':5,'target_first':True},}}},
	# 	{'exp_name': '7_1_32a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':10,'target_first':True},}}},
	# 	{'exp_name': '7_1_33a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'trajectory','env_kwargs': {'num_targets':-1,'target_first':True},}}},
	# ]
	# for trial in trial_configs:
	# 	trial['env_config'].update({'reward_type':'distance_target','phi':lambda d:0,})
	# 	trial['env_config'].update(action_map['trajectory'])
	# 	trial.update({'curriculum': True, 'lr': 1e-4})

	"""no noise inverse reward traj control"""
	trial_configs = [
		{'exp_name': '7_1_34', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'target',}}},
		{'exp_name': '7_1_34a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'target',}}},
		
		{'exp_name': '7_1_35', 'seed': 1000, 'env_config':{**r_config,**{'oracle': 'target',}}},
		{'exp_name': '7_1_35a', 'seed': 1001, 'env_config':{**r_config,**{'oracle': 'target',}}},
	]
	for trial in trial_configs:
		trial['env_config'].update({'noise_type':'none','reward_type':'distance_target','phi':lambda d:1/d,'env_kwargs':{'target_first':True},})
		trial['env_config'].update(action_map['trajectory'])
		trial.update({'curriculum': True, 'wrapper': MovingInit, 'lr': 1e-4})

	runners = [Runner.remote() for _i in range(len(trial_configs))]
	runners = [runner.run.remote(config) for runner,config in zip(runners,trial_configs)]
	runners = [ray.get(runner) for runner in runners]
