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
                 deterministic=False, random_latent=False):
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

    def get_action(self, obs):
        if obs['target1_reached'] and not self.target_switched:
            self.running_latent_sum = np.zeros(self.latent_size)
            self.running_latent_count = 0
            self.target_switched = True

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
                eps = th.normal(ptu.zeros(self.latent_size), 1) if self.sample else None
                single_trans_features = self.curr_vae.sample(encoder_input, eps=eps).detach().cpu().numpy()
                self.running_latent_sum += single_trans_features
                self.running_latent_count += 1
                pred_features = self.running_latent_sum / self.running_latent_count
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
        self.running_latent_sum = np.zeros(self.latent_size)
        self.running_latent_count = 0
        self.target_switched = False


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
