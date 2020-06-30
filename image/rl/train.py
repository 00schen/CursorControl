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

@ray.remote(num_cpus=1,num_gpus=1)
class Runner:
	def run(self,config):
		env_config = config['env_config']
		logdir = os.path.join(args.local_dir,config['exp_name'])
		os.makedirs(logdir, exist_ok=True)
		env = TargetRegion if config['curriculum'] else PreviousN
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

	# """joint, lr 1e-4"""
	# trial_configs = [
	# 	{'exp_name': '6_30_1', 'seed': 1000,},
	# 	{'exp_name': '6_30_1a', 'seed': 1001,},
	# ]
	# for trial in trial_configs:
	# 	trial['env_config'] = {**r_config,**{'env_kwargs': {'target_first': False}, 'oracle': 'trajectory'}}
	# 	trial['env_config'].update(action_map['joint'])
	# 	trial.update({'relabel_all': True, 'curriculum': True, 'lr': 1e-4})

	# """joint, lr 3e-4"""
	# trial_configs = [
	# 	{'exp_name': '6_30_2', 'seed': 1000,},
	# 	{'exp_name': '6_30_2a', 'seed': 1001,},
	# ]
	# for trial in trial_configs:
	# 	trial['env_config'] = {**r_config,**{'env_kwargs': {'target_first': False}, 'oracle': 'trajectory'}}
	# 	trial['env_config'].update(action_map['joint'])
	# 	trial.update({'relabel_all': True, 'curriculum': True, 'lr': 3e-4})

	# """traj, lr 1e-4"""
	# trial_configs = [
	# 	{'exp_name': '6_30_3', 'seed': 1000,},
	# 	{'exp_name': '6_30_3a', 'seed': 1001,},
	# ]
	# for trial in trial_configs:
	# 	trial['env_config'] = {**r_config,**{'env_kwargs': {'target_first': False}, 'oracle': 'trajectory'}}
	# 	trial['env_config'].update(action_map['trajectory'])
	# 	trial.update({'relabel_all': True, 'curriculum': True, 'lr': 1e-4})

	# """traj, lr 3e-4"""
	# trial_configs = [
	# 	{'exp_name': '6_30_4', 'seed': 1000,},
	# 	{'exp_name': '6_30_4a', 'seed': 1001,},
	# ]
	# for trial in trial_configs:
	# 	trial['env_config'] = {**r_config,**{'env_kwargs': {'target_first': False}, 'oracle': 'trajectory'}}
	# 	trial['env_config'].update(action_map['trajectory'])
	# 	trial.update({'relabel_all': True, 'curriculum': True, 'lr': 3e-4})

	"""joint, lr 1e-4, target oracle"""
	trial_configs = [
		{'exp_name': '6_30_5', 'seed': 1000,},
		{'exp_name': '6_30_5a', 'seed': 1001,},
	]
	for trial in trial_configs:
		trial['env_config'] = {**r_config,**{'env_kwargs': {'target_first': False}, 'oracle': 'target'}}
		trial['env_config'].update(action_map['joint'])
		trial.update({'relabel_all': True, 'curriculum': True, 'lr': 1e-4})

	runners = [Runner.remote() for _i in range(len(trial_configs))]
	runners = [runner.run.remote(config) for runner,config in zip(runners,trial_configs)]
	runners = [ray.get(runner) for runner in runners]
