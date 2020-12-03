import numpy as np

class FollowerPolicy:
	def get_action(self,obs):
		recommend = obs[:-6]
		if np.count_nonzero(recommend):
			self.action_index = np.argmax(recommend)
		action = np.zeros(6)
		action[self.action_index] = 1
		return action,{}
	def reset(self):
		self.action_index = 0