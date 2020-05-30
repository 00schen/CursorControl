# import matplotlib.pyplot as plt
import gym
import time
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import assistive_gym

from stable_baselines.sac import MlpPolicy
from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines import SAC
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.common import callbacks

from baselines.common.running_mean_std import RunningMeanStd
import numpy as np

import argparse

# time_now = time.strftime('%Y-%m-%d-%H-%M', time.localtime())


parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--env_name', help='gym name of environment')
parser.add_argument('--path_name', help='subfolder in logs to save to')
parser.add_argument('--lr', default='1e-4', help='learning rate')
args = parser.parse_args()
env_name = args.env_name

log_path = "../logs/sac/%s" % args.path_name
os.makedirs(log_path, exist_ok=True)

# gamma = .9

class norm_wrapper(gym.Env):
	def __init__(self,env):
		self.env = env
		self.ob_rms = RunningMeanStd(shape=env.observation_space.shape)
		self.observation_space = self.env.observation_space
		self.action_space = self.env.action_space

	def seed(self,seed=None):
		return self.env.seed(seed)
	
	def step(self,action):
		obs,r,done,info = self.env.step(action)
		obs = self._obfilt(obs)
		return obs,r,done,info

	def reset(self):
		# self.obs_rms = RunningMeanStd(shape=self.env.observation_space.shape)
		obs = self.env.reset()
		obs = self._obfilt(obs)
		return obs

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()

	def _obfilt(self, obs):
		self.ob_rms.update(np.array([obs]))
		obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + 1e-8), -10, 10)
		return obs

class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[64]*5,
                                           layer_norm=True,
                                           feature_extraction="mlp")

env = norm_wrapper(gym.make(env_name))
eval_env = norm_wrapper(gym.make(env_name))

env = Monitor(env, log_path)
model = SAC(MlpPolicy, env, learning_rate=float(args.lr), verbose=1, tensorboard_log=log_path+'/tensorboard')
callback = callbacks.CallbackList([
        callbacks.EvalCallback(eval_env, best_model_save_path=log_path, log_path=log_path),
        callbacks.CheckpointCallback(save_freq=int(5e4), save_path=log_path)])

time_steps = int(10e6)

model.learn(total_timesteps=time_steps,callback=callback,tb_log_name='run1')
print("Training Done")

# results_plotter.plot_results([log_path], time_steps, results_plotter.X_EPISODES, "SAC GoalControl")
# plt.show()