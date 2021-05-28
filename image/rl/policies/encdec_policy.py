import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import OneHotCategorical as TorchOneHot


class EncDecPolicy(PyTorchModule):
    def __init__(self, qf, features_keys, vae=None, logit_scale=-1, eps=0, incl_state=True, sample=False,
                 latent_size=None):
        super().__init__()
        self.vae = vae
        self.qf = qf
        self.features_keys = features_keys
        self.logit_scale = logit_scale  # currently does nothing
        self.eps = eps
        self.incl_state = incl_state
        self.sample = sample
        self.latent_size = latent_size
        if self.sample:
            assert self.latent_size is not None

    def get_action(self, obs):
        features = [obs[k] for k in self.features_keys]
        with th.no_grad():
            raw_obs = obs['raw_obs']
            if self.vae != None:
                if self.incl_state:
                    features.append(raw_obs)
                encoder_input = th.Tensor(np.concatenate(features)).to(ptu.device)
                eps = th.normal(ptu.zeros(self.latent_size), 1) if self.sample else None

                pred_features = self.vae.sample(encoder_input, eps=eps).detach()
                obs['latents'] = pred_features.cpu().numpy()
            else:
                pred_features = np.concatenate(features)
                obs['latents'] = pred_features

            q_values, ainfo = self.qf.get_action(raw_obs, pred_features)
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
