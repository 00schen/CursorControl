import numpy as np

class OverridePolicy:
	def __init__(self,policy):
		self.policy = policy
	def get_action(self,obs):
		recommend = obs[:-6]
		action,info = self.policy.get_action(obs)
		if np.count_nonzero(recommend):
			return recommend,info
		else:
			return action, info
	def reset(self):
		self.policy.reset()
