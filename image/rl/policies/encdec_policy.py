import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import (
    Delta
)
from rlkit.torch.networks.stochastic.distribution_generator import DistributionGenerator
import random


class EncDecPolicy(PyTorchModule):
    def __init__(self, policy, features_keys, vaes=None, incl_state=True, sample=False, latent_size=None,
                 deterministic=False, random_latent=False, window=None):
        super().__init__()

        self.vaes = vaes if vaes is not None else []
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
        self.curr_vae = None
        self.window = window if window is not None else 1
        self.past_factors = []

    def get_action(self, obs):
        features = [obs[k] for k in self.features_keys]
        with th.no_grad():
            raw_obs = obs['raw_obs']
            goal_set = obs.get('goal_set')

            if self.random_latent:
                pred_features = self.episode_latent.detach().cpu().numpy()
            elif len(self.vaes):
                if self.incl_state:
                    features.append(raw_obs)
                    if goal_set is not None:
                        features.append(goal_set.ravel())
                encoder_input = th.Tensor(np.concatenate(features)).to(ptu.device)
                mean, logvar = self.curr_vae.encode(encoder_input)
                self.past_factors.append((mean, logvar))

                self.past_factors = self.past_factors[-self.window:]
                mean, sigma_squared = self._product_of_gaussians(*zip(*self.past_factors))

                if self.sample:
                    posterior = th.distributions.Normal(mean, th.sqrt(sigma_squared))
                    pred_features = posterior.rsample()
                else:
                    pred_features = mean
                pred_features = pred_features.cpu().numpy()

            else:
                pred_features = np.concatenate(features)

            obs['latents'] = pred_features

            policy_input = [raw_obs, pred_features]
            if goal_set is not None:
                policy_input.insert(1, goal_set.ravel())
            action = self.policy.get_action(*policy_input)
            return action

    def reset(self):
        if self.random_latent:
            self.episode_latent = th.normal(ptu.zeros(self.latent_size), 1).to(ptu.device)
        self.policy.reset()
        if len(self.vaes):
            self.curr_vae = random.choice(self.vaes)
        self.past_factors = []

    def _product_of_gaussians(self, means, logvars):
        sigmas_squared = th.clamp(th.exp(th.stack(logvars)), min=1e-7)
        sigma_squared = 1. / th.sum(th.reciprocal(sigmas_squared), dim=0)
        mean = sigma_squared * th.sum(th.stack(means) / sigmas_squared, dim=0)
        return mean, sigma_squared

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
