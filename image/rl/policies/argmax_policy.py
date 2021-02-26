import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu

class ArgmaxPolicy(PyTorchModule):
	def __init__(self, qf, eps=0):
		super().__init__()
		self.qf = qf
		self.eps = eps

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf.parameters()).is_cuda:
			obs = obs.cuda()

		q_values, info = self.qf.get_action(obs)
		action = np.zeros(6)
		action[np.argmax(q_values)] = 1
		if np.random.rand() < self.eps:
			np.random.shuffle(action)
		return action, info

	def reset(self):
		pass