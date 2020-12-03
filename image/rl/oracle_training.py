##code not used

class LightSwitchTrainOracle(Agent):
	def __init__(self,env,qf1, qf2, epsilon=.8):
		super().__init__()
		self.env = env
		self.epsilon = epsilon
		self.policy = ArgmaxDiscretePolicy(
			qf1=qf1,
			qf2=qf2,
		)

	def get_action(self, obs):
		base_env = self.env.base_env
		rng = self.env.rng

		switches = np.array(base_env.switches)[np.nonzero(np.not_equal(base_env.target_string,base_env.current_string))]
		target_poses = np.array(base_env.target_pos)[np.nonzero(np.not_equal(base_env.target_string,base_env.current_string))]
		if len(target_poses) == 0:
			return np.array([1,0,0,0,0,0])

		if norm(target_poses[0]-base_env.tool_pos) < .12:
			action,_info = self.policy.get_action(obs)
		else:
			action = np.zeros(6)
			current_traj = [
				np.array((-1,0,0)),
				np.array((1,0,0)),
				np.array((0,-1,0)),
				np.array((0,1,0)),
				np.array((0,0,-1)),
				np.array((0,0,1)),
			][self.current_index]
			correct_traj = target_poses[0] - base_env.tool_pos
			cos = np.dot(current_traj,correct_traj)/(norm(current_traj)*norm(correct_traj))
			if cos < .5:
				traj = target_poses[0]-base_env.tool_pos
				axis = np.argmax(np.abs(traj))
				self.current_index = index = 2*axis+(traj[axis]>0)
				if rng.random() < self.epsilon:
					index = rng.integers(6)
				action[index] = 1
			else:
				action[self.current_index] = 1
		return action,{}
	def reset(self):
		self.current_index = 0


def train_oracle_factory(base):
	class TrainOracle(base):
		def __init__(self,config):
			super().__init__(config)
			# self.initial_oracle = UserModelOracle(self,**config['oracle_kwargs'])
			# self.initial_policy = TranslationPolicy(self,DemonstrationPolicy(self,lower_p=.8),**config)
			self.observation_space = spaces.Box(-10,10,
									(get_dim(self.observation_space)+3+3+3*3+3,))
		def step(self,action):
			obs,r,done,info = super().step(action)
			base_env = self.base_env
			bad_switch = np.logical_and(info['angle_dir'] != 0,
										np.logical_or(np.logical_and(info['angle_dir'] < 0, base_env.target_string == 1),
											np.logical_and(info['angle_dir'] > 0, base_env.target_string == 0))).astype(int)
			bad_contact = (info['bad_contact'] > 0)\
						or (np.count_nonzero(bad_switch)>0)
			info['bad_contact'] = bad_contact
			switch_pos,__ = p.getBasePositionAndOrientation(base_env.switches[0], physicsClientId=base_env.id)
			obs = np.concatenate([base_env.target_string,base_env.current_string,*base_env.target_pos,switch_pos,obs]).ravel()
			return obs,r,done,info
		def reset(self):
			# if self.rng.random() < 1/3:
			# 	def init_start_pos(self,og_init_pos):
			# 		switch_pos, switch_orient = p.getBasePositionAndOrientation(self.switches[1], physicsClientId=self.id)
			# 		init_pos, __ = p.multiplyTransforms(switch_pos, switch_orient, [0,.3,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
			# 		return init_pos
			# 	self.base_env.init_start_pos = MethodType(init_start_pos,self.base_env)
			# 	obs = super().reset()
			# 	angle = p.getJointStates(self.base_env.switches[0], jointIndices=[0], physicsClientId=self.base_env.id)[0][0]
			# 	p.resetJointState(self.base_env.switches[0], jointIndex=0, targetValue=-1-angle, physicsClientId=self.base_env.id)
			# elif self.rng.random() < 1/2:
			# 	def init_start_pos(self,og_init_pos):
			# 		switch_pos, switch_orient = p.getBasePositionAndOrientation(self.switches[2], physicsClientId=self.id)
			# 		init_pos, __ = p.multiplyTransforms(switch_pos, switch_orient, [0,.3,0], p.getQuaternionFromEuler([0,0,0]), physicsClientId=self.id)
			# 		return init_pos
			# 	self.base_env.init_start_pos = MethodType(init_start_pos,self.base_env)
			# 	obs = super().reset()
			# 	angle = p.getJointStates(self.base_env.switches[0], jointIndices=[0], physicsClientId=self.base_env.id)[0][0]
			# 	p.resetJointState(self.base_env.switches[0], jointIndex=0, targetValue=-1-angle, physicsClientId=self.base_env.id)
			# 	angle = p.getJointStates(self.base_env.switches[1], jointIndices=[0], physicsClientId=self.base_env.id)[0][0]
			# 	p.resetJointState(self.base_env.switches[1], jointIndex=0, targetValue=-1-angle, physicsClientId=self.base_env.id)
			# else:
			# 	obs = super().reset()

			# self.initial_oracle.reset()
			# self.initial_policy.reset()
			# obs,r,done,info = self.step(np.zeros(7))
			# bad_contact_found = False
			# for i in range(100):
			# 	self.recommend,_info = self.initial_oracle.get_action(obs,info)
			# 	if info['bad_contact']:
			# 		bad_contact_found = True
			# 		break
			# 	action,_info = self.initial_policy.get_action(obs)
			# 	obs,r,done,info = self.step(action)
			# 	self.timestep = 0
			# if not bad_contact_found:
			# 	return self.reset()
			# return obs

			obs = super().reset()
			base_env = self.base_env
			switch_pos,__ = p.getBasePositionAndOrientation(base_env.switches[0], physicsClientId=base_env.id)
			return np.concatenate([base_env.target_string,base_env.current_string,*base_env.target_pos,switch_pos,obs]).ravel()
	return TrainOracle

class train_li_oracle:
	def __init__(self,master_env,config):
		pass
	def _step(self,obs,r,done,info):
		r = 0
		target_string = obs[:3]
		current_string = obs[3:6]
		target_pos = obs[6:15].reshape((3,3))
		tool_pos = obs[-15:-12]
		target_indices = np.nonzero(np.not_equal(target_string,current_string))[0]
		if len(target_indices) > 0:
			# r -= min([norm(self.tool_pos-self.target_pos[i]) for i in target_indices])
			r -= 10*norm(tool_pos-target_pos[target_indices[0]])
		else:
			r -= 0
		for i in range(3):
			if target_string[i] == 0:
				r -= 250*abs(-.02 - info['angle_dir'][i])
			else:
				r -= 250*abs(.02 - info['angle_dir'][i])
		r -= 10*len(target_indices)
		print(obs[:6],info['angle_dir'],r)
		return obs,r,done,info
	def _reset(self,obs):
		return obs