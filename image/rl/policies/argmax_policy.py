import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu

class ArgmaxPolicy(PyTorchModule):
	def __init__(self, qf):
		super().__init__()
		self.qf = qf

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf.parameters()).is_cuda:
			obs = obs.cuda()

		with th.no_grad():
			_, q_values = self.qf(obs)
			action = F.one_hot(q_values.argmax().long(),6).flatten()
		return ptu.get_numpy(action), {}

	def reset(self):
		pass