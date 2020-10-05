from collections import OrderedDict

import numpy as np

from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.envs.env_utils import get_dim

from envs import rng
import torch as th
import torch.nn.functional as F
from collections import deque

class PavlovSubtrajReplayBuffer(ReplayBuffer):
	def __init__(
		self,
		max_num_traj,
		traj_max,
		env,
		obs_dim,
		pf,
	):
		self.pf = pf
		self.env = env
		action_dim = get_dim(env.action_space)
		observation_dim = obs_dim
		self._observation_dim = observation_dim
		self._action_dim = action_dim
		self._max_replay_buffer_size = max_num_traj
		self._current_obs_dim = env.current_obs_dim

		self._observations = np.zeros((max_num_traj, traj_max, observation_dim))
		# It's a bit memory inefficient to save the observations twice,
		# but it makes the code *much* easier since you no longer have to
		# worry about termination conditions.
		self._next_obs = np.zeros((max_num_traj, traj_max, observation_dim))
		self._actions = np.zeros((max_num_traj, traj_max, action_dim))
		# Make everything a 2D np array to make it easier for other code to
		# reason about the shape of the data
		self._rewards = np.zeros((max_num_traj, traj_max, 1))
		# self._terminals[i] = a terminal was received at time i
		# self._terminals = np.zeros((max_num_traj, traj_max, 1), dtype='uint8')
		# self._nonnoops = np.zeros((max_num_traj, traj_max, 1), dtype='uint8')
		# self._targets = np.zeros((max_num_traj, traj_max, 3),)
		# self._recommends = np.zeros((max_num_traj, traj_max, 1), dtype='uint8')
		self._successes = np.zeros((max_num_traj, traj_max, 1), dtype='uint8')
		self._lengths = np.zeros((max_num_traj, 1), dtype='int32')

		self._top = 0
		self._sample = 0
		self._index = 0
		self._valid_start_indices = set()
		self._val_valid_start_indices = set()
		self._add_counter = 0
		self._valid_indices = set()
		self._traj_max = traj_max

	def update_embeddings(self):
		obs_pf_hx = (th.zeros((2,self._top,self.pf.hidden_size)),th.zeros((2,self._top,self.pf.hidden_size)))
		next_pf_hx = (th.zeros((2,self._top,self.pf.hidden_size)),th.zeros((2,self._top,self.pf.hidden_size)))
		for i in range(self._traj_max):
			obs = th.as_tensor(self._observations[:self._top,[i],:self._current_obs_dim]).transpose(0,1)
			next_obs = th.as_tensor(self._next_obs[:self._top,[i],:self._current_obs_dim]).transpose(0,1)
			action = th.as_tensor(self._actions[:self._top,[i],:]).transpose(0,1)
			
			"""pf embeddings"""
			# obs_prediction = th.zeros((1,self._top,self._action_dim))
			# next_prediction = th.zeros((1,self._top,self._action_dim))
			# obs_preds,next_preds = [],[]
			# if next(self.pf.parameters()).is_cuda:
			# 	obs_prediction = obs_prediction.cuda()
			# 	next_prediction = next_prediction.cuda()
			# 	obs,next_obs,action = obs.cuda(),next_obs.cuda(),action.cuda()
			# 	obs_pf_hx = (obs_pf_hx[0].cuda(),obs_pf_hx[1].cuda())
			# 	next_pf_hx = (next_pf_hx[0].cuda(),next_pf_hx[1].cuda())
			# for j,oh_action in enumerate(1.0*F.one_hot(th.arange(0,self._action_dim),self._action_dim)):
			# 	oh_action = oh_action.repeat(1,self._top,1)
			# 	if next(self.pf.parameters()).is_cuda:
			# 		oh_action = oh_action.cuda()
			# 	obs_pred, _pf_hx = self.pf(th.cat((obs,oh_action),dim=2).float(),obs_pf_hx)
			# 	next_pred, _pf_hx = self.pf(th.cat((next_obs,oh_action),dim=2).float(),next_pf_hx)
			# # 	obs_prediction[...,j],next_prediction[...,j] = obs_pred.squeeze(),next_pred.squeeze()
			# 	obs_preds.append(obs_pred.squeeze())
			# 	next_preds.append(next_pred.squeeze())
			

			# self._observations[:self._top,i,-self.end[0]:-self.end[1]] = th.cat(obs_pf_hx,dim=2).squeeze().cpu().detach().numpy()
			# self._next_obs[:self._top,i,-self.end[0]:-self.end[1]] = th.cat(next_pf_hx,dim=2).squeeze().cpu().detach().numpy()
			# self._observations[:self._top,i,-self.end[1]:] = obs_prediction.squeeze().cpu().detach().numpy()
			# self._next_obs[:self._top,i,-self.end[1]:] = next_prediction.squeeze().cpu().detach().numpy()

			_pred, obs_pf_hx = self.pf(th.cat((obs,action),dim=2).float(),obs_pf_hx)
			_pred, next_pf_hx = self.pf(th.cat((next_obs,action),dim=2).float(),next_pf_hx)
			self._observations[:self._top,i,-self.pf.hidden_size:] = obs_pf_hx[0][0,...].squeeze().cpu().detach().numpy()
			self._next_obs[:self._top,i,-self.pf.hidden_size:] = next_pf_hx[0][0,...].squeeze().cpu().detach().numpy()

	def terminate_episode(self):
		pass

	def add_sample(self, observation, action, reward, next_observation,
				   terminal, env_info, **kwargs):
		pass

	def add_path(self, path):
		n_items = len(path["observations"])

		self._observations[self._sample,:n_items,:] = path['observations']
		# self._observations[self._sample,:n_items,:-self.pf.hidden_size] = path['observations']
		self._actions[self._sample,:n_items,:] = path['actions']
		self._rewards[self._sample,:n_items,:] = path['rewards']
		# self._terminals[self._sample,:n_items,:] = path['terminals']
		self._next_obs[self._sample,:n_items,:] = path['next_observations']
		# self._next_obs[self._sample,:n_items,:-self.pf.hidden_size] = path['next_observations']
		self._lengths[self._sample] = n_items
		print(path['rewards'].mean())

		# self._observations[self._sample,:n_items,-self.env.env.num_targets:] = np.array([np.arange(0,self.env.env.num_targets) == env_info['target_index'] for env_info in path['env_infos']])
		# self._next_obs[self._sample,:n_items,-self.env.env.num_targets:] = np.array([np.arange(0,self.env.env.num_targets) == env_info['target_index'] for env_info in path['env_infos']])
		
		# self._nonnoops[self._sample,:n_items] = np.array([not env_info['noop'] for env_info in path['env_infos'][1:]]+[False])[:,np.newaxis]
		# self._targets[self._sample,:n_items,:] = np.array([env_info['targets'][env_info['target_index']] for env_info in path['env_infos']])
		# self._targets[self._sample,:n_items] = np.array([env_info['target_index'] for env_info in path['env_infos']])[:,np.newaxis]
		# self._recommends[self._sample,:n_items] = np.array([np.argmax(env_info['recommend']) if np.count_nonzero(env_info['recommend']) else 6\
															# for env_info in path['env_infos'][1:]]+[6])[:,np.newaxis]
		self._successes[self._sample,:n_items,:] = np.array([env_info['task_success'] for env_info in path['env_infos']])[:,np.newaxis]


		# if self._add_counter % 10 != 0:
		# 	self._valid_start_indices.add(self._sample)
		# else:
		# 	self._val_valid_start_indices.add(self._sample)
		# if 1 in self._successes[self._sample,:n_items,:]:
		self._valid_start_indices.add(self._sample)
		self._add_counter += 1

		self._valid_indices.update([(self._sample,start_index) for start_index in range(n_items)])
		self._sample = (self._sample + 1) % self._max_replay_buffer_size
		self._top = min(self._top+1,self._max_replay_buffer_size)

	def random_traj_batch(self,num_traj):
		start_indices = np.array(list(self._valid_start_indices))[rng.choice(
			len(self._valid_start_indices),
			num_traj,
		)]
		batch = dict(
			observations=self._observations[start_indices],
			actions=self._actions[start_indices],
			rewards=self._rewards[start_indices],
			# terminals=self._terminals[start_indices],
			# next_observations=self._next_obs[start_indices],
			nonnoops=self._nonnoops[start_indices],
			recommends=self._recommends[start_indices],
			# targets=self._targets[start_indices],
			lengths=self._lengths[start_indices],
		)
		return batch

	def val_random_traj_batch(self,num_traj):
		start_indices = np.array(list(self._val_valid_start_indices))[rng.choice(
			len(self._val_valid_start_indices),
			num_traj,
		)]
		batch = dict(
			observations=self._observations[start_indices],
			actions=self._actions[start_indices],
			rewards=self._rewards[start_indices],
			# terminals=self._terminals[start_indices],
			# next_observations=self._next_obs[start_indices],
			nonnoops=self._nonnoops[start_indices],
			recommends=self._recommends[start_indices],
			# targets=self._targets[start_indices],
			lengths=self._lengths[start_indices],
		)
		return batch

	def random_batch(self,batch_size):
		indices = rng.choice(len(self._valid_indices),batch_size)
		index_pairs = np.array(list(self._valid_indices))
		sample_indices,traj_indices = index_pairs[indices,0],index_pairs[indices,1]
		batch = dict(
			observations=self._observations[sample_indices,traj_indices,:],
			actions=self._actions[sample_indices,traj_indices,:],
			rewards=self._rewards[sample_indices,traj_indices,:],
			# terminals=self._terminals[sample_indices,traj_indices,:],
			successes=self._successes[sample_indices,traj_indices,:],
			next_observations=self._next_obs[sample_indices,traj_indices,:],
		)
		return batch

	def num_steps_can_sample(self):
		return len(self._valid_start_indices)

	def get_diagnostics(self):
		return OrderedDict([
			('size', len(self._valid_start_indices))
		])

	def __getstate__(self):
		# Do not save self.replay_buffer since it's a duplicate and seems to
		# cause joblib recursion issues.
		return dict(
			observations = self._observations[:self._top],
			actions = self._actions[:self._top],
			rewards = self._rewards[:self._top],
			terminals = self._terminals[:self._top],
			next_obs = self._next_obs[:self._top],
			# inputs = self._inputs[:self._top],
			successes = self._successes[:self._top],

			observation_dim = self._observation_dim,
			action_dim = self._action_dim,
			current_obs_dim = self._current_obs_dim,
			pred_dim = self._pred_sizes,
			max_replay_buffer_size = self._max_replay_buffer_size,
			traj_max=self._traj_max,

			top = self._top,
			sample=self._sample,
			index=self._index,
			valid_start_indices=self._valid_start_indices,
			valid_indices = self._valid_indices,
		)

	def __setstate__(self, d):
		self._observation_dim = d["observation_dim"]
		self._action_dim = d["action_dim"]
		self._current_obs_dim = d['current_obs_dim']
		self._pred_sizes = d['pred_dim']
		self._max_replay_buffer_size = d["max_replay_buffer_size"]
		self._traj_max = d['traj_max']

		self._top = d['top']
		self._sample = d['sample']
		self._index = d['index']
		self._valid_start_indices = d['valid_start_indices']
		self._valid_indices = d['valid_indices']
		
		self._observations[:self._top] = d["observations"]
		self._next_obs[:self._top] = d["next_obs"]
		self._actions[:self._top] = d["actions"]
		self._rewards[:self._top] = d["rewards"]
		self._terminals[:self._top] = d["terminals"]
		self._successes[:self._top] = d["successes"]
		# self._inputs[:self._top] = d['inputs']
