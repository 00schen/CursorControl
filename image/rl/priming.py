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

@ray.remote(num_cpus=1,num_gpus=.16)
class PrimingRunner:
	def run(self,config):
		env_config = config['env_config']
		logdir = os.path.join(args.local_dir,config['exp_name'])
		os.makedirs(logdir, exist_ok=True)
		env = MovingEnd if config['curriculum'] else PreviousN
		env = make_vec_env(lambda: env(env_config),monitor_dir=logdir)
		env = VecNormalize(env,norm_reward=False)
		model = SAC(MlpPolicy, env, learning_rate=5e-5, train_freq=-1, n_episodes_rollout=1, gradient_steps=200,
					verbose=1,tensorboard_log=logdir)

		reward_name = SAMPLE_NAME[config['primer']]
		path = os.path.join(os.path.abspath(''),"replays",f"replay.{env_config['env_name'][0]}.{reward_name}")
		with open(path, "rb") as file_handler:
			buffer = pickle.load(file_handler)
		buffer.__class__ = PostProcessingReplayBuffer
		buffer.device = 'cuda'
		if config['prime_buffer']:
			model.replay_buffer = buffer
		else:
			model.replay_buffer.__class__ = PostProcessingReplayBuffer

		if config['bc']:
			for _i in tqdm(range(8000)):
				behavioural_clone(model.policy,buffer,env)

		stats = os.path.join(os.path.abspath(''),"replays",f"replay.{env_config['env_name'][0]}.stats.npz")\
					if config['primer']==NORM else None
		callback = CallbackList([
				RewardRolloutCallback(reward_type=config['primer'], replay_stats=stats),
				TensorboardCallback(curriculum=config['curriculum']),
				NormCheckpointCallback(save_freq=int(5e4), save_path=logdir),
				CheckpointCallback(save_freq=int(5e4), save_path=logdir),
			])
		time_steps = int(2e6)
		model.learn(total_timesteps=time_steps,callback=callback,tb_log_name=config['exp_name'])

def behavioural_clone(policy,replay_buffer,env):
	replay_data = replay_buffer.sample(128, env=env)

	actions_pi, _log_prob = policy.actor.action_log_prob(replay_data.observations)
	actor_loss = torch.square(actions_pi-replay_data.actions).mean()
	policy.actor.optimizer.zero_grad()
	actor_loss.backward()
	policy.actor.optimizer.step()
	print(actor_loss)

if __name__ == "__main__":
	ray.init(temp_dir='/tmp/ray_exp1')
	curriculum_default_config = {'num_obs': 5, 'oracle': 'trajectory', 'coop': False, 'end_early': False}
	env_config = deepcopy(default_config)
	env_config.update(curriculum_default_config)
	env_config.update(action_map['joint'])

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

	# trial_configs = [
	# 	{'exp_name': '6_26_0', 'primer': 4, 'env_config': deepcopy(f_config), 'curriculum': True, 'prime_buffer': True},
	# 	{'exp_name': '6_26_1', 'primer': 4, 'env_config': deepcopy(f_config), 'curriculum': True, 'prime_buffer': False},
	# 	{'exp_name': '6_26_2', 'primer': 4, 'env_config': deepcopy(f_config), 'curriculum': False, 'prime_buffer': True},
	# 	{'exp_name': '6_26_3', 'primer': 4, 'env_config': deepcopy(f_config), 'curriculum': False, 'prime_buffer': False},
	# ]

	trial_configs = [
		{'exp_name': '6_26_4', 'primer': 4, 'env_config': deepcopy(r_config), 'curriculum': False, 'prime_buffer': True, 'bc': True},
		{'exp_name': '6_26_5', 'primer': 4, 'env_config': deepcopy(r_config), 'curriculum': False, 'prime_buffer': True, 'bc': False},
		{'exp_name': '6_26_6', 'primer': 4, 'env_config': deepcopy(r_config), 'curriculum': False, 'prime_buffer': False, 'bc': True},
		{'exp_name': '6_26_7', 'primer': 4, 'env_config': deepcopy(r_config), 'curriculum': False, 'prime_buffer': False, 'bc': False},
	]

	runners = [Runner.remote() for _i in range(len(trial_configs))]
	runners = [runner.run.remote(config) for runner,config in zip(runners,trial_configs)]
	runners = [ray.get(runner) for runner in runners]
