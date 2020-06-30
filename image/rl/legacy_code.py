class BatchNoise():
	def __init__(self, action_space, dim, sd=.1, dropout=.3, lag=.85, batch=1, include=(0,1,2)):
		self.sd = sd
		self.batch = batch
		self.dim = dim
		self.lag = lag
		self.dropout = dropout

		self.action_space = action_space

		# random.seed(12345)
		self.noise = random.normal(np.repeat(np.identity(self.dim)[np.newaxis,...],self.batch,0), self.sd)
		self.lag_buffer = [deque([],10) for i in range(self.batch)]

		self.noises = [[self._add_grp,self._add_lag,self._add_dropout][i] for i in include]

	def _add_grp(self, action):
		return (action[:,np.newaxis,:]@self.noise)[:,0,:]

	def _add_dropout(self, action):
		return np.array([action_i if random.random() > self.dropout else self.action_space.sample() for action_i in action])

	def _add_lag(self, action):
		for buffer,act in zip(self.lag_buffer,action):
			buffer.append(act)
		return np.array([buffer.popleft() if random.random() > self.lag else buffer[0] for buffer in self.lag_buffer])

	def __call__(self,action):
		return reduce(lambda value,func: func(value), self.noises, action)

class VanillaPretrain:
	def __init__(self,config):
		if config['coop']:
			self.robot,self.human,self.norm = torch.load(
				os.path.join(parentname,'trained_models','ppo',f"{config['env_name']}JacoHuman-v0.pt"))
			self.robot_obs_size = config['obs_size']
		else:
			self.robot,self.norm = torch.load(
				os.path.join(parentname,'trained_models','ppo',f"{config['env_name']}Jaco-v0.pt"))
			self.human = None
			

	def predict(self,env,pred,obs):
		obs = env.target2obs(pred,obs)
		# obs = self._obfilt(obs)
		if self.human is None:
			obs = torch.tensor(obs,dtype=torch.float)

			with torch.no_grad():
				_value, action, _, _recurrent_hidden_states = self.robot.act(obs,
				torch.zeros(1, self.robot.recurrent_hidden_state_size), torch.ones(1,1), deterministic=True)
			action = action.numpy()[0,0,0]
		else:
			robot_obs = torch.tensor(obs[:self.robot_obs_size],dtype=torch.float)
			human_obs = torch.tensor(obs[self.robot_obs_size:],dtype=torch.float)

			with torch.no_grad():
				_value, robot_action, _, _recurrent_hidden_states = self.robot.act(robot_obs,
				torch.zeros(1, self.robot.recurrent_hidden_state_size), torch.ones(1,1), deterministic=True)
				_value, human_action, _, _recurrent_hidden_states = self.human.act(human_obs,
				torch.zeros(1, self.human.recurrent_hidden_state_size), torch.ones(1,1), deterministic=True)
			action = np.concatenate((robot_action.numpy()[0,0,0],human_action.numpy()[0,0,0]))

		return action

	def _obfilt(self, obs):
		obs = np.clip((obs - self.norm.mean) / np.sqrt(self.norm.var + 1e-8), -10, 10)
		return obs

class GymWrapper(Env):
	def __init__(self,config):
		self.coop = config['coop']
		env_name = config['env_name']+'Human' if config['coop'] else config['env_name']
		self.env = {
			"ScratchItch": assistive_gym.ScratchItchJacoEnv,
			"ScratchItchHuman": assistive_gym.ScratchItchJacoHumanEnv,
			"Feeding": assistive_gym.FeedingJacoEnv,
			"FeedingHuman": assistive_gym.FeedingJacoHumanEnv,
			"Laptop": assistive_gym.LaptopJacoEnv,
			"LaptopHuman": assistive_gym.LaptopJacoEnv,
			"LightSwitch": assistive_gym.LightSwitchJacoEnv,
			"LightSwitchHuman": assistive_gym.LightSwitchJacoEnv,
			"Reach": assistive_gym.ReachJacoEnv,
			"ReachHuman": assistive_gym.ReachJacoEnv
		}[env_name](**config['env_kwargs'])

		if config['action_type'] in ['target', 'trajectory']:
			self.pretrain = config['pretrain'](config)
		joint_action = lambda action,obs: action
		target_action = lambda pred,obs: self.pretrain.predict(self.env,pred,obs)
		trajectory_action = lambda traj,obs: target_action(self.env.tool_pos+tf.clip_by_norm(traj,.05).numpy(),obs)
		self.translate_action = {
			'joint': joint_action,
			'target': target_action,
			'trajectory': trajectory_action,
		}[config['action_type']]
		self.action_space = config['action_space']
		self.human_obs = np.array([])

	def step(self,action):
		action = (action[0],(action[1][0],np.concatenate((action[1][1],self.human_obs))))
		obs,r,done,info = self.env.step(self.translate_action(*action))
		if self.coop:
			self.human_obs = obs[self.env.obs_robot_len:]
		return obs,r,done,info

	def reset(self):
		obs = self.env.reset()
		if self.coop:
			self.human_obs = obs[self.env.obs_robot_len:]
		return obs

	def render(self,mode=None,**kwargs):
		self.env.render()
	def seed(self,value):
		self.env.seed(value)
	def close(self):
		self.env.close()

class SharedAutonomy(GymWrapper):
	def __init__(self,config):
		super().__init__(config)
		config['determiner'] = {
			'target': lambda: self.env.target_pos,
			'trajectory': lambda: self.env.target_pos - self.env.tool_pos,
			'discrete': lambda: self.env.target_num,
		}[config['oracle']]
		self.oracle = Oracle(config)

	def step(self,action):
		obs,r,done,info = super().step((action,self.og_obs))

		oracle_obs = self.oracle.predict(obs)
		self.og_obs = oracle_obs['obs']
		obs = np.concatenate((*oracle_obs['obs'],oracle_obs['action']))
		return obs,r,done,info

	def reset(self):
		obs = super().reset()
		self.oracle.reset()
		obs = self.oracle.predict(obs)
		self.og_obs = obs['obs']
		obs = np.concatenate((*obs['obs'],obs['action']))
		return obs

class Data:
	def __init__(self,pool_size=int(5e4),batch_size=256):
		self.data = {'X':deque([],pool_size),'Y':deque([],pool_size)}
		self.batch_size = batch_size

	def update(self,X,Y):
		self.data['X'].extend(X)
		self.data['Y'].extend(Y)

	def __getitem__(self,key):
		assert key == 'X' or key == 'Y', "Data only has keys 'X' and 'Y'."
		return np.array(self.data[key])

	def batch(self):
		idx = np.random.choice(range(len(self.data['X'])),self.batch_size,replace=False)
		return np.array(self.data['X'])[idx],np.array(self.data['Y'])[idx]

class Supervised:
	def __init__(self,config):
		obs_size,pred_step = config['sa_obs_size'],config['pred_step']
		self.norm = RunningMeanStd(shape=(obs_size,))
		self.data = Data(batch_size=64)
		self.pred_step = pred_step
		self.obs_history = deque([],200)
		self.model = make_train_LSTM((200-pred_step,obs_size))
		self.model.compile(Adam(lr=5e-3),'mse')
		self.pred_model = make_pred_LSTM(obs_size)
		self.save_path = config['save_path']
		self.train_iter = 0

	def predict(self,obs):
		obs = obs.flatten()
		self.obs_history.append(obs)

		sample = self._obfilt(obs)
		if not self.train_iter % 50:
			self.pred_model.load_weights(self.save_path,by_name=True)
		hidden_state = self.pred_model.predict(sample.reshape((1,1,-1)))
		return np.concatenate((obs,hidden_state.flatten()))

	def train(self):
		if len(self.obs_history) == self.obs_history.maxlen:
			self._update_data(np.array(self.obs_history)[np.newaxis,...])
		obs,target = self.data.batch()
		sample = self._obfilt(obs)
		self.model.train_on_batch(sample,target)
		self.train_iter += 1
		if not self.train_iter % 50:
			self.model.save_weights(self.save_path,save_format='h5')

	def reset(self):
		self.obs_history.clear()
		self.pred_model.load_weights(self.save_path,by_name=True)
	
	def pretrain(self,initial_batch,epochs=1):
		self._update_data(initial_batch)
		obs,target = self.data['X'],self.data['Y']
		obs = self._obfilt(obs)
		self.model.fit(obs,target,
						epochs=1,verbose=1,
						)
		self.model.save_weights(self.save_path,save_format='h5')

	"""regular LSTM"""
	def _update_data(self,X):
		obs = X[:,:-self.pred_step,:]
		target = X[:,self.pred_step:,-3:]

		self.norm.update(X.reshape((-1,X.shape[-1])))
		self.data.update(obs,target)

	def _obfilt(self, obs):
		obs = np.clip((obs - self.norm.mean) / np.sqrt(self.norm.var + 1e-8), -10, 10)
		return obs

def make_train_LSTM(input_shape,channels=64):
	episode_input = tf.keras.Input(shape=input_shape,name='episode')

	x = LSTM(channels,return_sequences=True,name='lstm1')(episode_input)
	recommendation_pred = Dense(3,name='dense')(x)
	model = tf.keras.Model(inputs=[episode_input],outputs=[recommendation_pred])
	return model

def make_pred_LSTM(input_shape,channels=64):
	episode_input = tf.keras.Input(batch_shape=(1,1,input_shape),name='episode')

	x,state_h,state_c = LSTM(channels,stateful=True,return_state=True,name='lstm1')(episode_input)
	# model = tf.keras.Model(inputs=[episode_input],outputs=[state_c])
	recommendation_pred = Dense(3,name='dense')(x)
	model = tf.keras.Model(inputs=[episode_input],outputs=[recommendation_pred])
	return model

def make_train_seq2seq(obs_shape,channels=64):
	obs_input = tf.keras.Input(shape=(None,obs_shape),name='observation')
	traj_input = tf.keras.Input(shape=(None,3),name='trajectory')
	
	x,state_h,state_c = LSTM(channels,return_sequences=True,return_state=True,name='encoder1')(obs_input)
	
	x = LSTM(channels,return_sequences=True,name='decoder1')(traj_input,initial_state=[state_h,state_c])
	
	recommendation_pred = Dense(3,name='dense')(x)
	model = tf.keras.Model(inputs=[obs_input,traj_input],outputs=[recommendation_pred])
	return model

class RadialCurriculum(PreviousN):
	def __init__(self,config):
		super().__init__(config)
		self.radius_limits = (0.01,config['radius_max'])
		self.success_radius = self.radius_limits[1]
		self.wait_time = 0

	def step(self,action):
		obs,r,done,info = super().step(action)
		if self.success_radius > self.radius_limits[0]: # up to certain distance, count just getting nearer the target
			success = norm(self.env.tool_pos-self.env.target_pos) < self.success_radius
			self.env.task_success += success
			r += success

		return obs,r,done,info

	def reset(self):
		self.wait_time += 1
		if self.wait_time > 20 and np.mean(self.success_count) > .5:
			self.success_radius -= .01
			self.wait_time = 0
		return super().reset()

class MovingEnd(PreviousN):
	def __init__(self,config):
		super().__init__(config)
		self.t = .1
		self.wait_time = 0

	def step(self,action):
		obs,r,done,info = super().step(action)
		
		if len(self.sub_targets):
			success = np.any(norm(self.env.tool_pos-self.sub_targets,axis=1) < .02)
			self.env.task_success += success
			r += success
		return obs,r,done,info

	def reset(self):
		obs = super().reset()
		self.wait_time += 1
		if self.wait_time > 20 and np.mean(self.success_count) > .5:
			self.t = min(self.t + .01, 1)
			self.wait_time = 0

		self.sub_targets = np.array([(1-t)*self.env.tool_pos + t*self.env.target_pos 
										for t in np.arange(self.t,1,.01)])

		return obs

class ResidualLearning(PreviousN):
	def __init__(self,config):
		super().__init__(config)
		self.action_gamma = config['action_gamma']

	def step(self,action):
		old_tool_pos = self.env.tool_pos
		obs,r,done,info = super().step(action)
		r += -self.action_gamma*cosine(self.env.tool_pos-old_tool_pos,np.mean(self.prev_inputs,axis=0))
		self.prev_inputs.append(obs[-3:])
		return obs,r,done,info

	def reset(self):
		obs = super().reset()
		self.prev_inputs = deque([],10)
		self.prev_inputs.append(obs[-3:])
		return obs