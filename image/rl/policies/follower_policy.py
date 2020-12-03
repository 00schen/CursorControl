import numpy as np

class FollowerPolicy:
	def __init__(self,oracle_size):
		self.oracle_size = oracle_size
	def get_action(self,obs):
		recommend = obs[-self.oracle_size:]
		if np.count_nonzero(recommend):
			self.action_index = np.argmax(recommend)
		action = np.zeros(6)
		action[self.action_index] = 1
		return action,{}
	def reset(self):
		self.action_index = 0