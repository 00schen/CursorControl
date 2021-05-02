import numpy as np
import torch as th
from .encdec_policy import EncDecPolicy
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import OneHotCategorical as TorchOneHot

class RandTargetPolicy(EncDecPolicy):
	def __init__(self, *args, env, eps=.1, **kwargs):
		super().__init__(*args, **kwargs)
		self.env
		self.eps = eps

	def get_action(self, obs):
		features = [obs[k] for k in self.features_keys]
		with th.no_grad():
			raw_obs = obs['raw_obs']
			if np.random.random() > self.eps:
				if self.encoder != None:
					# features.append(np.zeros(self.encoder.input_size-sum([len(f) for f in features])))
					pred_features,_ = self.encoder.get_action(*features)
				else:
					pred_features = np.concatenate(features)
			else:
				pred_features = self.env.base_env.get_random_target()
			
			q_values, ainfo = self.qf.get_action(raw_obs,pred_features[:3])
			q_values = ptu.tensor(q_values)
			if self.logit_scale != -1:
				action = TorchOneHot(logits=self.logit_scale*(q_values-q_values.mean())/(1e-8+q_values.std())).sample().flatten().detach()
			else:
				action = F.one_hot(q_values.argmax().long(),6).flatten()
			
		return ptu.get_numpy(action), ainfo

	def reset(self):
		pass