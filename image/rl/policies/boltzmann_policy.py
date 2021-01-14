import numpy as np
import torch as th
from rlkit.torch.distributions import Distribution
from rlkit.torch.core import PyTorchModule
from rlkit.torch.distributions import OneHotCategorical as TorchOneHot

class OneHotCategorical(Distribution,TorchOneHot):
	def rsample_and_logprob(self):
		s = self.sample()
		log_p = self.log_prob(s)
		return s, log_p

class BoltzmannPolicy(PyTorchModule):
	def __init__(self, qf1, qf2, logit_scale=100):
		super().__init__()
		self.qf1 = qf1
		self.qf2 = qf2
		self.logit_scale = logit_scale

	def get_action(self, obs):
		if isinstance(obs,np.ndarray):
			obs = th.from_numpy(obs).float()
		if next(self.qf1.parameters()).is_cuda:
			obs = obs.cuda()

		with th.no_grad():
			# q_values = th.min(self.qf1(obs),self.qf2(obs))
			q_values = self.qf1(obs)
			action = OneHotCategorical(logits=self.logit_scale*q_values).sample().flatten().detach()
		return action.cpu().numpy(), {}

	def reset(self):
		pass