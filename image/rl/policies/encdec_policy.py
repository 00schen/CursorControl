import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import OneHotCategorical as TorchOneHot

class EncDecPolicy(PyTorchModule):
	def __init__(self,  qf, features_keys, encoder=None, logit_scale=-1, eps=0):
		super().__init__()
		self.encoder = encoder
		self.qf = qf
		self.features_keys = features_keys
		self.logit_scale = logit_scale
		self.eps = eps

	def get_action(self, obs):
		features = [obs[k] for k in self.features_keys]
		with th.no_grad():
			raw_obs = obs['raw_obs']
			if self.encoder != None:
				pred_features = self.encoder.sample(th.Tensor(np.concatenate(features)).to(ptu.device)).detach()
				obs['latents'] = pred_features.cpu().numpy()
			else:
				pred_features = np.concatenate(features)
				obs['latents'] = pred_features

			q_values, ainfo = self.qf.get_action(raw_obs,pred_features[:3])
			q_values = ptu.tensor(q_values)
			if np.random.rand() > self.eps:
				action = F.one_hot(q_values.argmax().long(), 6).flatten()

			else:
				action = ptu.zeros(6)
				action[np.random.randint(6)] = 1

			# if self.logit_scale != -1:
			# 	action = TorchOneHot(logits=self.logit_scale * q_values).sample().flatten().detach()
			# 	# action = TorchOneHot(logits=self.logit_scale*(q_values-q_values.mean())/(1e-8+q_values.std())).sample().flatten().detach()
			# else:
			# 	action = F.one_hot(q_values.argmax().long(),6).flatten()

		return ptu.get_numpy(action), ainfo

	def reset(self):
		pass
