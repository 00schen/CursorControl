import os
import argparse

import numpy as np
import numpy.random as random
import torch

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

parser = argparse.ArgumentParser(description='Sequence Modeling - Velocity Controlled 2D Simulation')
parser.add_argument('--file_append', default=0,
                    help='If running multiple instances, which one to start at')

load_dir = 'trained_models/ppo'

env_name = 'ScratchItchJaco-v0'

class Noise():
  def __init__(self, env, sd, dim):
    self.SD = sd
    self.DIM = dim

    self.env = env

    self.noise = random.normal(np.identity(self.DIM),self.SD)
    self.lag_buffer = []

  def _add_grp(self, action):
    return self.noise@action

  def _add_dropout(self, action):
    return action if random.random() > .1\
      else self.env.action_space.sample()

  def _add_lag(self, action):
    self.lag_buffer.append(action)
    return self.lag_buffer.pop(0) if random.random() > .1 else self.lag_buffer[0]

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
    
env = make_vec_envs(env_name, 1002, 1, None, None,
                    add_timestep=False, device='cpu', allow_early_resets=False,
                    setup_function=customize)

# We need to use the same statistics for normalization as used in training
actor_critic, ob_rms = torch.load(os.path.join(load_dir, env_name + ".pt"))

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

data = []
for i in range(int(1e6)):
    obs = env.reset()
    noise = Noise(env,.05,env.action_space.shape[0])
    done,target = False,None
    episode = []
    while not done:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=True)
        prev_obs = obs[0]

        obs, reward, done, info = env.step(action)
        masks.fill_(0.0 if done else 1.0)
        stored_action = noise.get_noise(action.numpy().flatten())
        episode.append([*prev_obs[:7],*prev_obs[13:],*stored_action])
        target = info[0]['target']

    episode.append([*target,*([0]*(env.observation_space.shape[0]+env.action_space.shape[0]-6-3))])
    data.append(episode)
    print("{} episodes done.".format(i + 1))
    if not (i+1) % int(100):
      np.save("data/scratch_itch_data_{}".format(i+1+4*int(5e4)), data)

    data = []


