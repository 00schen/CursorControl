import os
import argparse

import numpy as np
import numpy.random as random
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--seed', default=12345, help='Also the folder name data is saved under')
parser.add_argument('--batch', default=5, help='Number of environments to run in parallel')
parser.add_argument('--compute',default='cpu', help='Whether to use cpu or gpu')
args = parser.parse_args()

model_dir = 'trained_models/ppo'
save_dir = str(args.seed)
os.makedirs(save_dir, exist_ok=True)

env_name = 'ScratchItchJaco-v0'

class Noise():
	def __init__(self, env, sd, dim, batch_size):
		self.SD = sd
		self.DIM = dim

		self.env = env
		self.batch = batch_size

		self.noise = random.normal(
			np.repeat(np.identity(self.DIM).reshape((1,self.DIM,self.DIM)),repeats=batch_size,axis=0),
			self.SD)
		self.lag_buffer = [[]]*batch_size

	def _add_grp(self, action):
		return (action.reshape((self.batch,1,-1))@self.noise).reshape(self.batch,-1)

	def _add_dropout(self, action):
		return np.array([action[i] if random.random() > .1 \
			else self.env.action_space.sample() for i in range(len(action))]) 
			

	def _add_lag(self, action):
		[buffer.append(action_i) for buffer,action_i in zip(self.lag_buffer,action)]
		return np.array([buffer.pop(0) if random.random() > .1 else buffer[0] for buffer in self.lag_buffer])

	def get_noise(self,action):
		return self._add_lag(self._add_dropout(self._add_grp(action)))

def customize(env):
    temp = env.__class__.step
    def add_target(self,action):
        obs,r,d,info = temp(self,action)
        info.update({"target":self.target_pos})
        return obs,r,d,info
    env._max_episode_steps = 500
    env.__class__.step = add_target
    
if __name__ == "__main__":
	batch,seed = int(args.batch),int(args.seed)
	np.random.seed(seed)
	env = make_vec_envs(env_name, seed, batch, None, None,
                    add_timestep=False, device=args.compute, allow_early_resets=False,
                    setup_function=customize)

	# We need to use the same statistics for normalization as used in training
	actor_critic, ob_rms = torch.load(os.path.join(model_dir, env_name + ".pt"))

	vec_norm = get_vec_normalize(env)
	if vec_norm is not None:
		vec_norm.eval()
		vec_norm.ob_rms = ob_rms

	data = []
	for i in range(int(1e2)//batch):
		obs = env.reset()
		recurrent_hidden_states = torch.zeros(batch, actor_critic.recurrent_hidden_state_size)
		masks = torch.zeros(batch, 1)
		noise = Noise(env,.05,env.action_space.shape[0],batch)
		done,target = np.zeros(batch), None
		episode = []
		while not np.all(done): # Scratch Itch will only finish because of gym cutoff
			with torch.no_grad():
				_value, action, _, recurrent_hidden_states = actor_critic.act(
					obs, recurrent_hidden_states, masks, deterministic=True)
			prev_obs = obs.numpy()
			stored_action = noise.get_noise(action.numpy())

			obs, reward, done, info = env.step(action)
			masks = torch.from_numpy(done.reshape((-1,1)))
			
			step = np.concatenate((prev_obs[:,:7],prev_obs[:,13:],stored_action),axis=1).reshape((batch,1,-1))
			episode = np.concatenate((episode,step),axis=1) if len(episode) else step
			target = [info_i['target'] for info_i in info]
		
		episode = np.concatenate((episode,
			np.concatenate((np.array(target),
				np.zeros((batch,env.observation_space.shape[0]+env.action_space.shape[0]-6-3))),
				axis=1).reshape((batch,1,-1))),axis=1)
		data = np.concatenate((data,episode),axis=0) if len(data) else episode
		print("{} episodes done.".format((i + 1)*batch))
		if not (i+1) % int(10):
			np.save(save_dir+'/scratch_itch_data_{}'.format(i+1), data)

			data = []


