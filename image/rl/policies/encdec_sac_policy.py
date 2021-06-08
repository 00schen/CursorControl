import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from rlkit.torch.distributions import (
    Delta
)
from rlkit.torch.core import elem_or_tuple_to_numpy
from rlkit.torch.sac.policies.base import TorchStochasticPolicy


class EncDecSACPolicy(PyTorchModule):
    def __init__(self, policy, features_keys, vae=None, incl_state=True, sample=False, latent_size=None,
                 deterministic=False):
        super().__init__()
        self.vae = vae
        self.policy = policy
        if deterministic:
            self.policy = EncDecMakeDeterministic(self.policy)
        self.features_keys = features_keys
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
                # print(encoder_input.shape,[(key,feature.shape) for key,feature in zip(self.features_keys,features)])
                pred_features = self.vae.sample(encoder_input, eps=eps).detach().cpu().numpy()
            else:
                pred_features = np.concatenate(features)

            obs['latents'] = pred_features
            action = self.policy.get_action(raw_obs, pred_features)
            return action

    def reset(self):
        self.policy.reset()


class EncDecMakeDeterministic(PyTorchModule):
    def __init__(
            self,
            policy,
    ):
        super().__init__()
        self.policy = policy

    def forward(self, *args, **kwargs):
        dist = self.policy.forward(*args, **kwargs)
        return Delta(dist.mle_estimate())

    def get_action(self, *obs_np):
        return self.policy.get_action(*obs_np)

    def get_actions(self, *obs_np):
        return self.policy.get_actions()

    def reset(self):
        self.policy.reset()
