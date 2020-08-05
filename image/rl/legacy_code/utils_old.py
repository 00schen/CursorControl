"""Ray Helpers"""
# from ray.rllib.agents import sac
# class CustomTrainable(sac.SACTrainer):
# 	def _save(self, checkpoint_dir):
# 		checkpoint_path = os.path.join(checkpoint_dir,
# 									f"checkpoint-{self.iteration}.h5")
# 		self.get_policy().model.action_model.save_weights(checkpoint_path,save_format='h5')

# 		os.makedirs(os.path.join(checkpoint_dir,"norm"), exist_ok=True)
# 		self.workers.local_worker().env.save(os.path.join(checkpoint_dir,"norm","norm"))
# 		return checkpoint_path

class RayPolicy:
	def __init__(self,obs_size,action_size,layer_sizes,path):
		model = self.model = keras.models.Sequential()
		model.add(Dense(layer_sizes[0],input_shape=(obs_size,)))
		for size in layer_sizes[1:]:
			model.add(Dense(size))
		model.add(Dense(action_size*2))
		model.load_weights(path)

	def predict(self,obs):
		obs = obs.reshape((1,-1))
		stats = self.model.predict(obs)[0]
		mean, log_std = np.split(stats, 2)
		log_std = np.clip(log_std, -20, 2)
		std = np.exp(log_std)
		action = np.random.normal(mean, std)
		return np.clip(np.tanh(action), -1, 1)

def norm_factory(base):
	class NormEnv(base):
		def __init__(self,**kwargs):
			super().__init__(**kwargs)
			self.norm = RunningMeanStd(shape=self.observation_space.shape)

		def step(self,action):
			obs,r,done,info = super().step(action)
			self.norm.update(obs[np.newaxis,...])
			obs = self._obfilt(obs)
			return obs,r,done,info

		def reset(self):
			obs = super().reset()
			self.norm.update(obs[np.newaxis,...])
			obs = self._obfilt(obs)
			return obs

		def _obfilt(self,obs):
			return (obs-self.norm.mean)/(1e-8+np.sqrt(self.norm.var))
		def save(self,path):
			with open(path, "wb") as file_handler:
				pickle.dump(self.norm, file_handler)
		def load(self,path):
			with open(path, "rb") as file_handler:
				self.norm = pickle.load(file_handler)
	return NormEnv

class PostProcessingReplayBuffer(ReplayBuffer):
	def reset_sample(self):
		self.sample_batch = {'obs':[], 'next_obs':[], 'action':[], 'reward':[], 'done':[], 'info':[]}
	def add(self, obs, next_obs, action, reward, done, info):
		self.sample_batch['obs'].append(obs)
		self.sample_batch['next_obs'].append(next_obs)
		self.sample_batch['action'].append(action)
		self.sample_batch['reward'].append(reward)
		self.sample_batch['done'].append(done)
		self.sample_batch['info'].append(info)
	def extend(self, *args, **kwargs):
		for data in zip(*args):
			ReplayBuffer.add(self,*data)
	def confirm(self):
		self.extend(self.sample_batch['obs'],
					self.sample_batch['next_obs'],
					self.sample_batch['action'],
					self.sample_batch['reward'],
					self.sample_batch['done'],
					self.sample_batch['info'])

class RewardRolloutCallback(BaseCallback):
	def __init__(self, relabel_all=False, verbose=0):
		super().__init__(verbose)
		self.relabel_all = relabel_all
	def _on_step(self):
		return True
	def _on_rollout_start(self):
		self.model.replay_buffer.reset_sample()
	def _on_rollout_end(self):
		batch = self.model.replay_buffer.sample_batch
		if not self.relabel_all:
			if batch['info'][-1][0]['task_success'] > 0:
				batch['reward'] += np.array([[info_[0]['diff_distance']] for info_ in batch['info']])
		else:
			batch['reward'] += np.array([[info_[0]['diff_distance']] for info_ in batch['info']])
	
		self.model.replay_buffer.confirm()

"""Pretrain"""
class TransitionFeeder:
	def __init__(self,config):
		obs_data,r_data,done_data,targets_data = config['obs_data'],config['r_data'],config['done_data'],config['targets_data']
		self.data = zip(obs_data,r_data,done_data,targets_data)
		self.env = config['env'].env.env
		self.env.reset()
		self.action_space = config['env'].action_space
		self.blank_obs = config['blank_obs']
	def reset(self):
		obs,r,done,targets = next(self.data,([],[],[],[]))
		self.obs,self.r,self.done,self.targets = iter(obs),iter(r),iter(done),iter(targets)
		return next(self.obs,None)
	def step(self,action):
		obs,r,done = next(self.obs,None),next(self.r,0),next(self.done,True)
		self.env.targets = next(self.targets,None)
		info = {'noop':not np.count_nonzero(obs[-6:])}

		if self.blank_obs:
			obs[:-6] = 0

		return obs,r,done,info

IndexReplayBufferSamples = namedtuple("IndexReplayBufferSamples",
	["observations","actions","next_observations", "dones", "rewards", "indices"])
class IndexReplayBuffer(ReplayBuffer):
	def __init__(self, buffer_size, observation_space, action_space, device = 'cpu', n_envs = 1):
		super().__init__(buffer_size, observation_space, action_space, device, n_envs)
		self.indices = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)		
	def add(self, obs, next_obs, action, reward, done, index=None):
		if index is not None:
			self.indices[self.pos] = np.array(index).copy()
		super().add(obs, next_obs, action, reward, done)
	def _get_samples(self,batch_inds,env):
		data = (self._normalize_obs(self.observations[batch_inds, 0, :], env),
				self.actions[batch_inds, 0, :],
				self._normalize_obs(self.next_observations[batch_inds, 0, :], env),
				self.dones[batch_inds],
				self._normalize_reward(self.rewards[batch_inds], env),
				self.indices[batch_inds])
		return IndexReplayBufferSamples(*tuple(map(self.to_torch, data)))

def pretrain(config,env,policy):
	if not config['bad_demonstrations']:
		config['obs_data'],action_data,config['r_data'],config['done_data'],index_data,config['targets_data'] =\
			np.load(os.path.join(config['og_path'],f"{config['env_name'][:2]}_demo.npz"),allow_pickle=True).values()
	else:
		config['obs_data'],action_data,config['r_data'],config['done_data'],index_data,config['targets_data'] =\
			np.load(os.path.join(config['og_path'],f"{config['env_name'][:2]}_demo1.npz"),allow_pickle=True).values()

	config['env'] = env.envs[0]
	t_env = window_factory(TransitionFeeder)(config)
	env.envs[0] = t_env
	buffer = IndexReplayBuffer(int(1e6), t_env.observation_space, t_env.action_space, device=policy.device, n_envs=1)

	obs = env.reset()
	action_data,index_data = iter(action_data),iter(index_data)
	for i in range(len(config['obs_data'])):
		action_ep,index_ep = iter(next(action_data,"act_seq_done")),next(index_data,"index_done")		
		done = False
		while not done:
			action = [next(action_ep,"action_done")]
			next_obs,r,done,info = env.step(action)	
			if not done:
				buffer.add(obs,next_obs,action,r,done,index_ep)	
				obs = next_obs
			else:
				prev_obs = obs
				obs = next_obs
				next_obs = [info[0]['terminal_observation']]
				buffer.add(prev_obs,next_obs,action,r,done,index_ep)

	with trange(int(buffer.size()*10/config['batch_size'])) as t:
		for gradient_step in t:
			replay_data = buffer.sample(config['batch_size'], env=env)
			logits,_kwargs = policy.get_action_dist_params(replay_data.observations)
			actor_loss = th.nn.CrossEntropyLoss()(logits,replay_data.indices.squeeze().long())
			with th.no_grad():
				actions = policy(replay_data.observations)
				accuracy = actions.eq(replay_data.indices.squeeze()).float().mean()
			logger.record('pretraining/accuracy',accuracy)
			logger.record('pretraining/log-loss',actor_loss)
			t.set_postfix(accuracy=accuracy.item(),loss=actor_loss.item())

			policy.optimizer.zero_grad()
			actor_loss.backward()
			policy.optimizer.step()

"""railrl"""
from railrl.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
class PathAdaptLoader(DictToMDPPathLoader):
	def load_path(self, path, replay_buffer, obs_dict=None):
		replay_buffer.env.adapt_path(path)
		super().load_path(path, replay_buffer, obs_dict)

def window_adapt(self,path):
	obs_iter = iter(path['observations']+[path['next_observations'][-1]])
	done_iter = iter(path['terminals'])
	info_iter = iter(path['env_infos'])
	history = deque(np.zeros(self.history_shape),self.history_shape[0])
	is_nonnoop = deque([False]*self.history_shape[0],self.history_shape[0])
	prev_nonnoop = deque(np.zeros(self.nonnoop_shape),self.nonnoop_shape[0])
	new_path = {'observations':[],'next_observations':[]}

	history.append(next(obs_iter))
	obs = np.concatenate((np.ravel(history),np.ravel(prev_nonnoop),))
	done = False
	while not done:
		new_path['observations'].append(obs)

		if len(history) == self.history_shape[0] and is_nonnoop[0]:
			prev_nonnoop.append(history[0])
		history.append(next(obs_iter))
		info = next(info_iter)
		info['adapt'] = False
		is_nonnoop.append(info['noop'])
		done = next(done_iter)

		obs = np.concatenate((np.ravel(history),np.ravel(prev_nonnoop),))
		new_path['next_observations'].append(obs)
	
	path.update(new_path)

def adapt_factory(base,adapt_funcs):
	class PathAdapter(base):
		def step(self,action):
			obs,r,done,info = super().step(action)
			info['adapt'] = False
			return obs,r,done,info
		def adapt_path(self,path):
			if path['env_infos'][0].get('adapt',True):
				return reduce(lambda value,func:func(self,value), adapt_funcs, path)
			return path
	return PathAdapter

from railrl.misc.eval_util import create_stats_ordered_dict, get_stat_in_paths
def logger_factory(base):
	class StatsLogger(base):
		def get_diagnostics(self,paths):
			statistics = OrderedDict()

			"""success"""
			success_per_step = get_stat_in_paths(paths, 'env_infos', 'task_success')
			success_per_ep = [np.count_nonzero(s) > 0 for s in success_per_step]
			statistics.update(create_stats_ordered_dict('success',success_per_ep,exclude_max_min=True,))	
			
			"""distance"""
			distance_per_step = get_stat_in_paths(paths, 'env_infos', 'distance_to_target')
			min_distance = [np.amin(s) for s in distance_per_step]
			init_distance = [s[0] for s in distance_per_step]
			final_distance = [s[-1] for s in distance_per_step]
			statistics.update(create_stats_ordered_dict('min_distance',min_distance,))
			statistics['init_distance'] = np.mean(init_distance)
			statistics['final_distance'] = np.mean(final_distance)

			"""cos_error"""
			cos_error_per_step = get_stat_in_paths(paths, 'env_infos', 'cos_error')
			statistics.update(create_stats_ordered_dict('cos_error',cos_error_per_step,))

			"""noop"""
			noop_per_step = get_stat_in_paths(paths, 'env_infos', 'noop')
			statistics.update(create_stats_ordered_dict('noop',noop_per_step,exclude_max_min=True,))

			return statistics
	return StatsLogger

railrl_class = lambda env_name, adapt_funcs: adapt_factory(logger_factory(default_class(env_name)),adapt_funcs)

from railrl.torch.sac.awac_trainer import AWACTrainer
from railrl.torch.networks import ConcatMlp
import railrl.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
from railrl.core.logging import add_prefix
from railrl.torch.core import np_to_pytorch_batch
class PavlovTrainer(AWACTrainer):
	def __init__(self,env,policy,qf1,qf2,target_qf1,target_qf2,buffer_policy=None,
				penalty=-1,af_lr=3e-4,optimizer_class=optim.Adam,**kwargs):
		super().__init__(env,policy,qf1,qf2,target_qf1,target_qf2,buffer_policy,**kwargs)
		self.penalty = penalty
		obs_dim = env.observation_space.low.size
		action_dim = env.action_space.low.size
		self.af1 = ConcatMlp(input_size=obs_dim + action_dim,output_size=2,hidden_sizes=[256]*3,)
		self.af2 = ConcatMlp(input_size=obs_dim + action_dim,output_size=2,hidden_sizes=[256]*3,)
		self.af_criterion = nn.CrossEntropyLoss(weight=th.tensor([1.,4.]).cuda())
		self.af1_optimizer = optimizer_class(self.af1.parameters(),lr=af_lr,)
		self.af2_optimizer = optimizer_class(self.af2.parameters(),lr=af_lr,)

	def test_from_torch(self, batch):
		obs = batch['observations']
		actions = batch['actions']
		inputs = batch['inputs'].flatten()

		a1_pred = self.af1(obs, actions)
		a2_pred = self.af2(obs, actions)
		af1_loss = self.af_criterion(a1_pred, inputs)
		af2_loss = self.af_criterion(a2_pred, inputs)

		self.eval_statistics['validation/AF1 Loss'] = np.mean(ptu.get_numpy(af1_loss))
		self.eval_statistics['validation/AF2 Loss'] = np.mean(ptu.get_numpy(af2_loss))
		super().test_from_torch(batch)

	def train_from_torch(self, batch, train=True, pretrain=False,):
		rewards = batch['rewards']
		terminals = batch['terminals']
		obs = batch['observations']
		actions = batch['actions']
		next_obs = batch['next_observations']
		inputs = batch['inputs'].flatten()
		weights = batch.get('weights', None)
		if self.reward_transform:
			rewards = self.reward_transform(rewards)

		if self.terminal_transform:
			terminals = self.terminal_transform(terminals)
		"""
		Policy and Alpha Loss
		"""
		dist = self.policy(obs)
		new_obs_actions, log_pi = dist.rsample_and_logprob()
		policy_mle = dist.mle_estimate()

		if self.use_automatic_entropy_tuning:
			alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
			alpha = self.log_alpha.exp()
		else:
			alpha_loss = 0
			alpha = self.alpha

		"""
		QF Loss
		"""
		q1_pred = self.qf1(obs, actions)
		q2_pred = self.qf2(obs, actions)
		# Make sure policy accounts for squashing functions like tanh correctly!
		next_dist = self.policy(next_obs)
		new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
		target_q_values = th.min(
			self.target_qf1(next_obs, new_next_actions),
			self.target_qf2(next_obs, new_next_actions),
		) - alpha * new_log_pi.reshape((-1,1))

		q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
		qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
		qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

		"""
		AF Loss
		"""
		a1_pred = self.af1(obs, actions)
		a2_pred = self.af2(obs, actions)
		af1_loss = self.af_criterion(a1_pred, inputs.long())
		af2_loss = self.af_criterion(a2_pred, inputs.long())

		"""
		Policy Loss
		"""
		qf1_new_actions = self.qf1(obs, new_obs_actions)
		qf2_new_actions = self.qf2(obs, new_obs_actions)
		q_new_actions = th.min(
			qf1_new_actions,
			qf2_new_actions,
		)
		af1_pred = self.af1(obs, actions)
		af2_pred = self.af2(obs, actions)

		# Advantage-weighted regression
		if self.awr_use_mle_for_vf:
			v1_pi = self.qf1(obs, policy_mle)
			v2_pi = self.qf2(obs, policy_mle)
			v_pi = th.min(v1_pi, v2_pi)
		else:
			if self.vf_K > 1:
				vs = []
				for i in range(self.vf_K):
					u = dist.sample()
					q1 = self.qf1(obs, u)
					q2 = self.qf2(obs, u)
					v = th.min(q1, q2)
					# v = q1
					vs.append(v)
				v_pi = th.cat(vs, 1).mean(dim=1)
			else:
				# v_pi = self.qf1(obs, new_obs_actions)
				v1_pi = self.qf1(obs, new_obs_actions)
				v2_pi = self.qf2(obs, new_obs_actions)
				v_pi = th.min(v1_pi, v2_pi)

		if self.awr_sample_actions:
			u = new_obs_actions
			if self.awr_min_q:
				q_adv = q_new_actions
			else:
				q_adv = qf1_new_actions
			af1_new_actions = self.af1(obs, new_obs_actions)
			af2_new_actions = self.af2(obs, new_obs_actions)
			input_adv = th.min(th.argmax(af1_new_actions,dim=1),th.argmax(af2_new_actions,dim=1),)
		else:
			u = actions
			if self.awr_min_q:
				q_adv = th.min(q1_pred, q2_pred)
			else:
				q_adv = q1_pred
			input_adv = th.min(th.argmax(af1_pred,dim=1),th.argmax(af2_pred,dim=1),)

		policy_logpp = dist.log_prob(u)

		beta = self.beta_schedule.get_value(self._n_train_steps_total)

		if self.normalize_over_state == "advantage":
			score = q_adv - v_pi
			if self.mask_positive_advantage:
				score = th.sign(score)
			score += self.penalty*input_adv.reshape((-1,1))
		else:
			error

		if self.clip_score is not None:
			score = th.clamp(score, max=self.clip_score)

		if self.weight_loss and weights is None:
			if self.normalize_over_batch == True:
				weights = F.softmax(score / beta, dim=0)
			elif self.normalize_over_batch == "whiten":
				adv_mean = th.mean(score)
				adv_std = th.std(score) + 1e-5
				normalized_score = (score - adv_mean) / adv_std
				weights = th.exp(normalized_score / beta)
			elif self.normalize_over_batch == "exp":
				weights = th.exp(score / beta)
			elif self.normalize_over_batch == "step_fn":
				weights = (score > 0).float()
			elif self.normalize_over_batch == False:
				weights = score
			else:
				error
		weights = weights[:, 0]

		policy_loss = alpha * log_pi.mean()

		if self.use_awr_update and self.weight_loss:
			policy_loss = policy_loss + self.awr_weight * (-policy_logpp * len(weights)*weights.detach()).mean()
		elif self.use_awr_update:
			policy_loss = policy_loss + self.awr_weight * (-policy_logpp).mean()

		if self.use_reparam_update:
			policy_loss = policy_loss + self.reparam_weight * (-q_new_actions).mean()

		policy_loss = self.rl_weight * policy_loss
		if self.compute_bc:
			train_policy_loss, train_logp_loss, train_mse_loss, _ = self.run_bc_batch(self.demo_train_buffer, self.policy)
			policy_loss = policy_loss + self.bc_weight * train_policy_loss

		"""
		Update networks
		"""
		if self._n_train_steps_total % self.policy_update_period == 0 and self.update_policy:
			self.policy_optimizer.zero_grad()
			policy_loss.backward()
			self.policy_optimizer.step()

		if self._n_train_steps_total % self.q_update_period == 0:
			self.qf1_optimizer.zero_grad()
			qf1_loss.backward()
			self.qf1_optimizer.step()

			self.qf2_optimizer.zero_grad()
			qf2_loss.backward()
			self.qf2_optimizer.step()

		self.af1_optimizer.zero_grad()
		af1_loss.backward()
		self.af1_optimizer.step()

		self.af2_optimizer.zero_grad()
		af2_loss.backward()
		self.af2_optimizer.step()

		"""
		Soft Updates
		"""
		if self._n_train_steps_total % self.target_update_period == 0:
			ptu.soft_update_from_to(
				self.qf1, self.target_qf1, self.soft_target_tau
			)
			ptu.soft_update_from_to(
				self.qf2, self.target_qf2, self.soft_target_tau
			)

		"""
		Save some statistics for eval
		"""
		if self._need_to_update_eval_statistics:
			self._need_to_update_eval_statistics = False
			"""
			Eval should set this to None.
			This way, these statistics are only computed for one batch.
			"""
			policy_loss = (log_pi - q_new_actions).mean()

			self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
			self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
			self.eval_statistics['AF1 Loss'] = np.mean(ptu.get_numpy(af1_loss))
			self.eval_statistics['AF2 Loss'] = np.mean(ptu.get_numpy(af2_loss))
			self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
				policy_loss
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q1 Predictions',
				ptu.get_numpy(q1_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q2 Predictions',
				ptu.get_numpy(q2_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Q Targets',
				ptu.get_numpy(q_target),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'A1 Predictions',
				ptu.get_numpy(a1_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'a2 Predictions',
				ptu.get_numpy(a2_pred),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Log Pis',
				ptu.get_numpy(log_pi),
			))
			policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
			self.eval_statistics.update(policy_statistics)
			self.eval_statistics.update(create_stats_ordered_dict(
				'Advantage Weights',
				ptu.get_numpy(weights),
			))
			self.eval_statistics.update(create_stats_ordered_dict(
				'Advantage Score',
				ptu.get_numpy(score),
			))

			if self.use_automatic_entropy_tuning:
				self.eval_statistics['Alpha'] = alpha.item()
				self.eval_statistics['Alpha Loss'] = alpha_loss.item()

			if self.compute_bc:
				test_policy_loss, test_logp_loss, test_mse_loss, _ = self.run_bc_batch(self.demo_test_buffer, self.policy)
				self.eval_statistics.update({
					"bc/Train Logprob Loss": ptu.get_numpy(train_logp_loss),
					"bc/Test Logprob Loss": ptu.get_numpy(test_logp_loss),
					"bc/Train MSE": ptu.get_numpy(train_mse_loss),
					"bc/Test MSE": ptu.get_numpy(test_mse_loss),
					"bc/train_policy_loss": ptu.get_numpy(train_policy_loss),
					"bc/test_policy_loss": ptu.get_numpy(test_policy_loss),
				})
			if self.train_bc_on_rl_buffer:
				_, buffer_train_logp_loss, _, _ = self.run_bc_batch(
					self.replay_buffer.train_replay_buffer,
					self.buffer_policy)

				_, buffer_test_logp_loss, _, _ = self.run_bc_batch(
					self.replay_buffer.validation_replay_buffer,
					self.buffer_policy)
				buffer_dist = self.buffer_policy(obs)
				kldiv = th.distributions.kl.kl_divergence(dist, buffer_dist)

				_, train_offline_logp_loss, _, _ = self.run_bc_batch(
					self.demo_train_buffer,
					self.buffer_policy)

				_, test_offline_logp_loss, _, _ = self.run_bc_batch(
					self.demo_test_buffer,
					self.buffer_policy)

			if self.validation_qlearning:
				train_data = self.replay_buffer.validation_replay_buffer.random_batch(self.bc_batch_size)
				train_data = np_to_pytorch_batch(train_data)
				obs = train_data['observations']
				next_obs = train_data['next_observations']
				# goals = train_data['resampled_goals']
				train_data['observations'] = obs # torch.cat((obs, goals), dim=1)
				train_data['next_observations'] = next_obs # torch.cat((next_obs, goals), dim=1)
				self.test_from_torch(train_data)

		self._n_train_steps_total += 1

	@property
	def networks(self):
		nets = [
			self.policy,
			self.qf1,
			self.qf2,
			self.target_qf1,
			self.target_qf2,
			self.af1,
			self.af2,
		]
		if self.buffer_policy:
			nets.append(self.buffer_policy)
		return nets

class _PavlovReplayBuffer(EnvReplayBuffer):
	def __init__(self,max_replay_buffer_size,env,env_info_sizes=None):
		super().__init__(max_replay_buffer_size,env,env_info_sizes)
		self._inputs = np.zeros((max_replay_buffer_size, 1))

	def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
		self._inputs[self._top] = not kwargs['env_info'].get('noop',True)
		super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)

	def random_batch(self, batch_size):
		indices = np.random.randint(0, self._size, batch_size)
		batch = dict(
			observations=self._observations[indices],
			actions=self._actions[indices],
			rewards=self._rewards[indices],
			terminals=self._terminals[indices],
			next_observations=self._next_obs[indices],
			inputs=self._inputs[indices],
		)
		for key in self._env_info_keys:
			assert key not in batch.keys()
			batch[key] = self._env_infos[key][indices]
		return batch

class PavlovReplayBuffer(_PavlovReplayBuffer):
	def __init__(self,max_replay_buffer_size,env,env_info_sizes=None):
		super().__init__(max_replay_buffer_size,env,env_info_sizes)
		self.noop_replay_buffer = _PavlovReplayBuffer(max_replay_buffer_size,env,env_info_sizes)

	def add_sample(self, observation, action, reward, terminal, next_observation, **kwargs):
		if kwargs['env_info'].get('noop',True):
			self.noop_replay_buffer.add_sample(observation, action, reward, terminal, next_observation, **kwargs)
		else:
			super().add_sample(observation, action, reward, terminal, next_observation, **kwargs)

	def random_batch(self, batch_size):
		if self._size >= batch_size//2:
			input_batch = super().random_batch(batch_size//2)
			noop_batch = self.noop_replay_buffer.random_batch(batch_size-batch_size//2)
		else:
			input_batch = super().random_batch(self._size)
			noop_batch = self.noop_replay_buffer.random_batch(batch_size-self._size)
		batch = dict(
			observations=np.concatenate((input_batch['observations'],noop_batch['observations'])),
			actions=np.concatenate((input_batch['actions'],noop_batch['actions'])),
			rewards=np.concatenate((input_batch['rewards'],noop_batch['rewards'])),
			terminals=np.concatenate((input_batch['terminals'],noop_batch['terminals'])),
			next_observations=np.concatenate((input_batch['next_observations'],noop_batch['next_observations'])),
			inputs=np.concatenate((input_batch['inputs'],noop_batch['inputs'])),
		)
		return batch

	def num_steps_can_sample(self):
		return super().num_steps_can_sample() + self.noop_replay_buffer.num_steps_can_sample()