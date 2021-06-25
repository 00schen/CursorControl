import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import (
    Delta
)
from rlkit.torch.networks.stochastic.distribution_generator import DistributionGenerator


class EncDecPolicy(PyTorchModule):
    def __init__(self, policy, features_keys, vae=None, incl_state=True, sample=False, latent_size=None,
                 deterministic=False, random_latent=False):
        super().__init__()
        self.vae = vae
        self.policy = policy
        if deterministic:
            assert isinstance(policy, DistributionGenerator)
            self.policy = EncDecMakeDeterministic(self.policy)
        self.features_keys = features_keys
        self.incl_state = incl_state
        self.sample = sample
        self.latent_size = latent_size
        if self.sample:
            assert self.latent_size is not None
        self.random_latent = random_latent
        self.episode_latent = None

    def get_action(self, obs):
        features = [obs['goal_obs']]
        with th.no_grad():
            base_obs = obs['base_obs']

            if self.random_latent:
                pred_features = self.episode_latent.detach().cpu().numpy()
            elif self.vae != None:
                if self.incl_state:
                    features.append(base_obs)
                encoder_input = th.Tensor(np.concatenate(features)).to(ptu.device)
                eps = th.normal(ptu.zeros(self.latent_size), 1) if self.sample else None
                pred_features = self.vae.sample(encoder_input, eps=eps).detach().cpu().numpy()
            else:
                pred_features = np.concatenate(features)

            obs['latents'] = pred_features

            policy_input = [base_obs, pred_features]
            action = self.policy.get_action(*policy_input)
            return action

    def reset(self):
        if self.random_latent:
            self.episode_latent = th.normal(ptu.zeros(self.latent_size), 1).to(ptu.device)
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
