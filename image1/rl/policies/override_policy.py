import numpy as np

class OverridePolicy:
	def __init__(self,policy,env,oracle_size):
		self.policy = policy
		self.env = env
		self.oracle_size = oracle_size
	def get_action(self,obs):
		recommend = self.env.recommend
		action,info = self.policy.get_action(obs)
		if np.count_nonzero(recommend):
			return recommend,info
		else:
			return action, info
	def reset(self):
		self.policy.reset()
