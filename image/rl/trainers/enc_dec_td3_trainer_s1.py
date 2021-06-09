from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.td3.td3 import TD3Trainer


class EncDecTD3Trainer(TD3Trainer):
    """
    Twin Delayed Deep Deterministic policy gradients
    """

    def __init__(
            self,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            target_policy,
            latent_size,
            target_policy_noise=0.2,
            target_policy_noise_clip=0.5,

            discount=0.99,
            reward_scale=1.0,

            policy_learning_rate=1e-3,
            qf_learning_rate=1e-3,
            encoder_learning_rate=1e-3,
            policy_and_target_update_period=2,
            tau=0.005,
            qf_criterion=None,
            optimizer_class=optim.Adam,
            vae=None,
            beta=0.01,
            sample=True,
    ):
        super().__init__(policy=policy,
                         qf1=qf1,
                         qf2=qf2,
                         target_qf1=target_qf1,
                         target_qf2=target_qf2,
                         target_policy=target_policy,
                         target_policy_noise=target_policy_noise,
                         target_policy_noise_clip=target_policy_noise_clip,
                         discount=discount,
                         reward_scale=reward_scale,
                         policy_learning_rate=policy_learning_rate,
                         qf_learning_rate=qf_learning_rate,
                         policy_and_target_update_period=policy_and_target_update_period,
                         tau=tau,
                         qf_criterion=qf_criterion,
                         optimizer_class=optimizer_class)

        self.vae = vae
        self.beta = beta
        self.sample = sample
        self.latent_size = latent_size

        if self.vae is not None:
            self.encoder_optimizer = optimizer_class(
                self.vae.encoder.parameters(),
                lr=encoder_learning_rate,
            )

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        curr_goal = batch['curr_goal']
        next_goal = batch['next_goal']
        curr_goal_set = batch.get('curr_goal_set')
        next_goal_set = batch.get('next_goal_set')

        batch_size = obs.shape[0]
        has_goal_set = curr_goal_set is not None

        eps = torch.normal(ptu.zeros((batch_size, self.latent_size)), 1) if self.sample else None

        if self.vae is not None:
            curr_latent, kl_loss = self.vae.sample(curr_goal, eps=eps, return_kl=True)
            next_latent = self.vae.sample(next_goal, eps=eps, return_kl=False)

            next_latent = next_latent.detach()

        else:
            curr_latent, next_latent = curr_goal, next_goal
            kl_loss = 0

        """
        Critic operations.
        """

        if has_goal_set:
            next_goal_set_flat = next_goal_set.reshape((batch_size, -1))
            next_policy_features = [next_obs, next_goal_set_flat, next_latent]
        else:
            next_policy_features = [next_obs, next_latent]

        next_actions = self.target_policy(*next_policy_features)
        noise = ptu.randn(next_actions.shape) * self.target_policy_noise
        noise = torch.clamp(
            noise,
            -self.target_policy_noise_clip,
            self.target_policy_noise_clip
        )
        noisy_next_actions = next_actions + noise

        if has_goal_set:
            next_qf_features = [next_obs, next_goal_set_flat, next_goal, noisy_next_actions]
        else:
            next_qf_features = [next_obs, next_goal, noisy_next_actions]

        target_q1_values = self.target_qf1(*next_qf_features)
        target_q2_values = self.target_qf2(*next_qf_features)
        target_q_values = torch.min(target_q1_values, target_q2_values)
        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        q_target = q_target.detach()

        if has_goal_set:
            curr_goal_set_flat = curr_goal_set.reshape((batch_size, -1))
            curr_qf_features = [obs, curr_goal_set_flat, curr_goal, actions]
        else:
            curr_qf_features = [obs, curr_goal, actions]

        q1_pred = self.qf1(*curr_qf_features)
        bellman_errors_1 = (q1_pred - q_target) ** 2
        qf1_loss = bellman_errors_1.mean()

        q2_pred = self.qf2(*curr_qf_features)
        bellman_errors_2 = (q2_pred - q_target) ** 2
        qf2_loss = bellman_errors_2.mean()

        """
        Update Networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        policy_actions = policy_loss = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            if has_goal_set:
                curr_policy_features = [obs, curr_goal_set_flat, curr_latent]
            else:
                curr_policy_features = [obs, curr_latent]

            policy_actions = self.policy(*curr_policy_features)

            if has_goal_set:
                new_qf_features = [obs, curr_goal_set_flat, curr_goal, policy_actions]
            else:
                new_qf_features = [obs, curr_goal, policy_actions]

            q_output = self.qf1(*new_qf_features)
            policy_loss = - q_output.mean()

            self.policy_optimizer.zero_grad()
            if self.vae is not None:
                self.encoder_optimizer.zero_grad()
            (policy_loss + self.beta * kl_loss).backward()
            self.policy_optimizer.step()
            if self.vae is not None:
                self.encoder_optimizer.step()

            ptu.soft_update_from_to(self.policy, self.target_policy, self.tau)
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.tau)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            if policy_loss is None:
                policy_actions = self.policy(obs)
                q_output = self.qf1(obs, policy_actions)
                policy_loss = - q_output.mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics['KL Loss'] = np.mean(ptu.get_numpy(
                kl_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 1',
                ptu.get_numpy(bellman_errors_1),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Bellman Errors 2',
                ptu.get_numpy(bellman_errors_2),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy Action',
                ptu.get_numpy(policy_actions),
            ))
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_policy,
            self.target_qf1,
            self.target_qf2,
        ]
        if self.vae is not None:
            nets.append(self.vae)
        return nets

    def get_snapshot(self):
        snapshot = dict(
            qf1=self.qf1,
            qf2=self.qf2,
            trained_policy=self.policy,
            target_policy=self.target_policy,
        )
        if self.vae is not None:
            snapshot['vae'] = self.vae
        return snapshot
