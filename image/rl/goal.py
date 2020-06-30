import gym
import time
import os
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"]="1"
import tensorflow as tf

from stable_baselines3.sac import MlpPolicy
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import BaseCallback,CallbackList
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import assistive_gym
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

from utils import *
from envs import *

parser = argparse.ArgumentParser()
parser.add_argument('--local_dir',default='~/share/image/rl/test',help='dir to save trials')
parser.add_argument('--exp_name',default='test',help='gym environment name')
parser.add_argument('--num_gpus', type=int, default=4, help='dir to save trials')
args, _ = parser.parse_known_args()
dirname = os.path.abspath('')

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

def run(config, reporter):
	env = make_vec_env(lambda: config['env_name']())
	env = VecNormalize(env,norm_reward=False)
	model = SAC(MlpPolicy, env, learning_rate=config['lr'], verbose=1, tensorboard_log=tune.get_trial_dir())

	if config['prime']:
		path = os.path.join(dirname,"replays",f"replay.R.sizes5")
		with open(path, "rb") as file_handler:
			norm = pickle.load(file_handler)
		model.replay_buffer = norm

	for _i in range(8000):
		behavioural_clone(model.policy,model.replay_buffer,env)

	class ReportCallback(BaseCallback):
		def __init__(self, verbose=0):
			super().__init__(verbose)
			self.mins = deque([],20)
		def _on_step(self):
			env = self.training_env.envs[0]
			self.min_dist = np.minimum(self.min_dist,np.linalg.norm(env.target_pos - env.tool_pos))
			if self.n_calls % 1000 == 0:
				tune.report(distance_target=np.mean(self.mins),
						timesteps_total=self.num_timesteps)
			return True
		def _on_rollout_start(self):
			self.min_dist = np.inf
		def _on_rollout_end(self):
			self.mins.append(self.min_dist)
	class TensorboardCallback(BaseCallback):
		def __init__(self, verbose=0):
			super().__init__(verbose)
			self.success_count = deque([0],20)
		def _on_step(self):
			env = self.training_env.envs[0]
			self.min_dist = np.minimum(self.min_dist,np.linalg.norm(env.target_pos - env.tool_pos))	
			self.logger.record('success_metric/success_rate', np.mean(self.success_count))
			if self.n_calls % 200 == 195:	
				self.logger.record('success_metric/min_distance', self.min_dist)
				self.logger.record('success_metric/final_distance', np.linalg.norm(env.target_pos - env.tool_pos))
				self.success_count.append(env.task_success>0)
			return True
		def _on_rollout_start(self):
			self.min_dist = np.inf
	class TuneCheckpointCallback(BaseCallback):
		def __init__(self, save_freq, verbose=0):
			super().__init__(verbose)
			self.save_freq = save_freq
		def _on_step(self):
			if self.n_calls % self.save_freq == 0:
				path = tune.get_trial_dir()
				self.training_env.save(os.path.join(path,f"norm.{self.num_timesteps}"))
				self.model.save(os.path.join(path,f"model.{self.num_timesteps}"))
				model.save_replay_buffer(os.path.join(path,f"replay.{self.num_timesteps}"))
			return True			
	callback = CallbackList([
		TuneCheckpointCallback(save_freq=int(5e4)),
		TensorboardCallback(),
		ReportCallback(),
	])

	model.learn(total_timesteps=config['time_limit'],callback=callback)

def behavioural_clone(policy,replay_buffer,env):
	replay_data = replay_buffer.sample(128, env=env)

	actions_pi, _log_prob = policy.actor.action_log_prob(replay_data.observations)
	actor_loss = torch.square(actions_pi-replay_data.actions).mean()
	policy.actor.optimizer.zero_grad()
	actor_loss.backward()
	policy.actor.optimizer.step()

if __name__ == "__main__":
	args = parser.parse_args()
	ray.init(temp_dir='/tmp/ray_exp', num_gpus=args.num_gpus)

	sched = AsyncHyperBandScheduler(
		time_attr="timesteps_total",
		metric="distance_target",
		mode="min",
		max_t=int(2.5e6),
		grace_period=int(5e5))

	space = {
		"lr": hp.uniform("lr",1e-5,1e-3),
		# "prime": hp.choice("prime",[True,False]),
	}
	current_best_params = [{
		"lr": 5e-5,
		# "prime": False
	}]
	algo = HyperOptSearch(
		space,
		metric="distance_target",
		mode="min",
		points_to_evaluate=current_best_params
		)

	config = {
		"env": timer_factory(assistive_gym.ReachJacoEnv),
		"time_limit": int(2.5e6),
		"prime": True
	}

	stop = {
		"timesteps_total": int(5e6),
	}

	results = tune.run(run, name= args.exp_name, local_dir=args.local_dir,
													 num_samples=50,
													 config=config, stop=stop,
													 search_alg=algo,
													 scheduler=sched, 
													 resources_per_trial={'cpu':1,'gpu':.5},
													#  trial_name_creator=trial_str_creator,
													 verbose=1)

	ray.shutdown()