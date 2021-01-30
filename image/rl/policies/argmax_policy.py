import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu

class ArgmaxPolicy(PyTorchModule):
	def __init__(self, qf1,):
		super().__init__()
		self.qf1 = qf1

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		with th.no_grad():
			# q_values = th.min(self.qf1(obs),self.qf2(obs))
			q_values = self.qf1(obs)
			action = F.one_hot(q_values.argmax().long(),6).flatten()
		return ptu.get_numpy(action), {}

	def reset(self):
		pass