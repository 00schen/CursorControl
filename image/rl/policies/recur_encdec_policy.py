import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import OneHotCategorical as TorchOneHot


class RecurEncDecPolicy(PyTorchModule):
	def __init__(self, encoder, qf, features_keys, logit_scale=-1):
		super().__init__()
		self.encoder = encoder
		self.qf = qf
		self.features_keys = features_keys
		self.logit_scale = logit_scale

	def get_action(self, obs):
		features = [obs[k] for k in self.features_keys]
		with th.no_grad():
			raw_obs = obs['raw_obs']
			features.append(np.zeros(self.encoder.input_size-sum([len(f) for f in features])))
			pred_features, (self.h,self.c) = self.encoder.get_action(raw_obs,*features, hx=(self.h,self.c))
			q_values, ainfo = self.qf.get_action(raw_obs,pred_features[:3])
			q_values = ptu.tensor(q_values)
			if self.logit_scale != -1:
				action = TorchOneHot(logits=self.logit_scale*(q_values-q_values.mean())/(1e-8+q_values.std())).sample().flatten().detach()
			else:
				action = F.one_hot(q_values.argmax().long(),6).flatten()
			
		return ptu.get_numpy(action), ainfo

	def reset(self):
		self.h,self.c = ptu.zeros((self.encoder.num_layers,1,self.encoder.hidden_size)),ptu.zeros((self.encoder.num_layers,1,self.encoder.hidden_size))