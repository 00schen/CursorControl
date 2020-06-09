import os
import argparse

import numpy as np

import assistive_gym
import gym
from gym import spaces

import ray
from tqdm import tqdm

from utils import VanillaPretrain,BatchNoise

parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--batch', default=25, type=int, help='Number of environments to run in parallel')
parser.add_argument('--env_name', default='ScratchItchJaco-v0', help='gym name of assistive environment')
args = parser.parse_args()

@ray.remote
class Sampler:
	def sample(self,env_name,seed,count):
		env_name0 = env_name[:-1]+'0'
		env_name1 = env_name[:-1]+'1'
		env = gym.make(env_name1)
		env.seed(seed)
		rng = np.random.default_rng(seed)
		agent = VanillaPretrain(env_name0)

		obs_data = []
		traj_data = []
		targets = []
		for i in tqdm(range(count)):
			obs_eps = []
			traj_eps = []
			target_eps = []

			obs = env.reset()
			targets.append(env.target_pos)

			done = False
			action = agent.predict(obs)
			while not done: # Scratch Itch will only finish because of gym cutoff
				trajectory = env.oracle2trajectory(action)
				obs_eps.append(obs)
				traj_eps.append(trajectory)

				obs,r,done,info = env.step(action) if rng.random() > .6 else env.step(env.action_space.sample())
				action = agent.predict(obs)

			obs_data.append(obs_eps)
			traj_data.append(traj_eps)

		return obs_data, traj_data, targets

if __name__ == "__main__":
	# ray.init(temp_dir='/tmp/ray_exp')

	# samplers = [Sampler.remote() for i in range(args.batch)]
	# samples = [samplers[i].sample.remote(args.env_name,1000+i,1500//args.batch) for i in range(args.batch)]
	# samples = [ray.get(sample) for sample in samples]
	# obs,trajectory,targets = zip(*samples)
	# obs,trajectory,targets = list(obs),list(trajectory),list(targets)
	# obs,trajectory,targets = np.concatenate(obs),np.concatenate(trajectory),np.concatenate(targets)

	# np.savez_compressed(f"{args.env_name[0]}.dropout_sample",obs=obs,trajectory=trajectory,targets=targets)
	
	# obs,trajectory,targets = np.load(f"{args.env_name[0]}.dropout_sample.npz").values()
	# act = np.copy(trajectory.transpose((1,0,2)))
	# noise = BatchNoise(spaces.Box(low=-.01*np.ones(3),high=.01*np.ones(3)),3,batch=act.shape[1])
	# for i in tqdm(range(len(act))):
	# 	act[i] = noise(act[i])
	# act = np.array(act).transpose((1,0,2))
	# np.savez_compressed(f"{args.env_name[0]}1.noised_trajectory",obs=obs,noised=act,unnoised=trajectory,targets=targets)

	obs,trajectory,targets = np.load(f"{args.env_name[0]}.dropout_sample.npz").values()
	act = np.copy(obs[...,7:10])
	act = np.array(act).transpose((1,0,2))
	noise = BatchNoise(spaces.Box(low=-1*np.ones(3),high=np.ones(3)),3,batch=act.shape[1])
	for i in tqdm(range(len(act))):
		act[i] = noise(act[i])
	act = np.array(act).transpose((1,0,2))
	np.savez_compressed(f"{args.env_name[0]}1.noised_trajectory",obs=obs,noised=act,unnoised=obs[...,7:10],targets=targets)

