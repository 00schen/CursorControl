import os
import argparse

import numpy as np
from numpy import savez_compressed as save

import torch

import assistive_gym
import gym

from utils import PretrainAgent
from stable_baselines.common import make_vec_env

parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--seed', default=12345, help='Also the folder name data is saved under')
# parser.add_argument('--batch', default=5, help='Number of environments to run in parallel')
parser.add_argument('--env_name', default='ScratchItchJaco-v0', help='gym name of assistive environment')
args = parser.parse_args()

model_dir = os.path.join('trained_models','ppo','ScratchItchJaco-v0.pt')
save_dir = os.path.join('samples',args.env_name)
os.makedirs(save_dir, exist_ok=True)

env_name = args.env_name

# class WrapperEnv(gym.Wrapper):
# 	def step(self,action):
# 		obs,r,done,info = self.env.step(action)
# 		info.update({'target':self.target_pos})
# 		return obs,r,done,info

if __name__ == "__main__":
	env = gym.make(env_name)
	
	agent = PretrainAgent(model_dir)
	
	obs_data = []
	act_data = []
	targets = []
	for i in range(int(2e3)):
		obs_eps = []
		act_eps = []
		target_eps = []

		obs = env.reset()
		targets.append(env.target_pos)

		done = False
		action = agent.predict(obs,done)
		while not done: # Scratch Itch will only finish because of gym cutoff
			obs_eps.append(obs)
			act_eps.append(action)
			
			obs,r,done,info = env.step(action)
			action = agent.predict(obs,done)
		
		obs_data.append(obs_eps)
		act_data.append(act_eps)
					
		print(f"{(i + 1)} episodes done.")
		if not len(obs_data) % 500:
			obs_data = np.array(obs_data)
			act_data = np.array(act_data)
			targets = np.array(targets)

			save(os.path.join(save_dir,f'{args.seed}_{(i + 1)//500}'),
				obs_data=obs_data,act_data=act_data,targets=targets)

			obs_data = []
			act_data = []
			targets = []



# if __name__ == "__main__":
# 	batch = int(args.batch)

# 	env = make_vec_env(args.env_name,n_envs=batch,seed=int(args.seed),wrapper_class=WrapperEnv)
	
# 	agent = PretrainAgent(model_dir)
	
# 	obs_data = []
# 	act_data = []
# 	targets = []
# 	for i in range(int(5e4)//batch):
# 		obs_eps = []
# 		act_eps = []
# 		target_eps = []

# 		obs = env.reset()
# 		done = False
# 		action = agent.predict(obs,done)
# 		while not done: # Scratch Itch will only finish because of gym cutoff
# 			obs_eps.append(obs[:,np.newaxis,:])
# 			act_eps.append(action[:,np.newaxis,:])
			
# 			obs,r,done,info = env.step(action)
# 			done = np.all(done)
# 			action = agent.predict(obs,done)
# 			target_eps = [info_i['target'] for info_i in info]
		
# 		obs_data.append(np.concatenate(obs_eps,axis=1))
# 		act_data.append(np.concatenate(act_eps,axis=1))
# 		targets.append(target_eps)
					
# 		print(f"{(i + 1)*batch} episodes done.")
# 		if len(obs_data) == 50:
# 			obs_data = np.concatenate(obs_data)
# 			act_data = np.concatenate(act_data)
# 			targets = np.concatenate(targets)

# 			save(os.path.join(save_dir,f'{args.seed}_{(i + 1)//500}'),
# 				obs_data=obs_data,act_data=act_data,targets=targets)

# 			obs_data = []
# 			act_data = []
# 			targets = []
