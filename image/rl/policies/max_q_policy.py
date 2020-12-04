import rlkit.torch.pytorch_util as ptu
import torch as th

class MaxQPolicy:
	def __init__(self,policy,qf):
		self.policy = policy
		self.qf = qf
	def get_action(self, obs):
		"""
		Used when sampling actions from the policy and doing max Q-learning
		"""
		with th.no_grad():
			state = ptu.from_numpy(obs.reshape(1, -1)).repeat(10, 1)
			dist = self.policy(state)
			actions = dist.sample()
			q1 = self.qf(state, actions)
			ind = q1.max(0)[1]
		return ptu.get_numpy(actions[ind]).flatten(),{}
	def reset(self):
		pass