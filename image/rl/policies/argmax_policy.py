import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu

class ArgmaxPolicy(PyTorchModule):
	def __init__(self, qf, features_keys):
		super().__init__()
		self.qf = qf
		self.features_keys = features_keys

	def get_action(self, obs):
		features = [obs[k] for k in ['raw_obs']+self.features_keys]
		with th.no_grad():
			q_values, ainfo = self.qf.get_action(*features)
			q_values = ptu.tensor(q_values)
			# q_values = self.qf(obs)
			action = F.one_hot(q_values.argmax().long(),6).flatten()
		return ptu.get_numpy(action), ainfo

	def reset(self):
		pass