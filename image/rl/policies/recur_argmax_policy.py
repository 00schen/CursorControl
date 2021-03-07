import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu

class RecurArgmaxPolicy(PyTorchModule):
	def __init__(self, qf):
		super().__init__()
		self.qf = qf

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf.parameters()).is_cuda:
			obs = obs.cuda()

		with th.no_grad():
			q_values, (self.h,self.c) = self.qf(obs[None,None,:], (self.h,self.c))
			q_values = q_values[0,0]
			action = F.one_hot(q_values.argmax().long(),6).flatten()
		return ptu.get_numpy(action), {}

	def reset(self):
		self.h,self.c = ptu.zeros((self.qf.num_layers,1,self.qf.hidden_size)),ptu.zeros((self.qf.num_layers,1,self.qf.hidden_size))

