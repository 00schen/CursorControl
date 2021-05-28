import torch as th
import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
from .dqn_trainer import DQNTrainer


class EncDecDDQNTrainer(DQNTrainer):
    def __init__(self,
                 qf, target_qf,
                 optimizer,
                 latent_size,
                 vae=None,
                 temp=1.0,
                 min_q_weight=1.0,
                 add_ood_term=-1,
                 beta=1,
                 sample=True,
                 **kwargs):
        super().__init__(qf, target_qf, optimizer, **kwargs)
        self.vae = vae
        self.temp = temp
        self.min_q_weight = min_q_weight
        self.add_ood_term = add_ood_term
        self.beta = beta
        self.sample = sample
        self.latent_size = latent_size

    def train_from_torch(self, batch):
        terminals = batch['terminals']
        actions = batch['actions']
        obs = batch['observations']
        next_obs = batch['next_observations']
        rewards = batch['rewards']
        # episode_success = batch['episode_success']
        curr_goal = batch['curr_goal']
        next_goal = batch['next_goal']

        loss = 0

        eps = th.normal(ptu.zeros((obs.size(0), self.latent_size)), 1) if self.sample else None

        if self.vae is not None:
            curr_latent, kl_loss = self.vae.sample(th.cat((curr_goal, obs), dim=1), eps=eps, return_kl=True)
            next_latent = self.vae.sample(th.cat((next_goal, next_obs), dim=1), eps=eps, return_kl=False)
            next_latent = next_latent.detach()

        else:
            curr_latent, next_latent = curr_goal, next_goal
            kl_loss = 0

        loss += self.beta * kl_loss

        curr_obs_features = [obs, curr_latent]
        next_obs_features = [next_obs, next_latent]

        """
        Q loss
        """
        best_action_idxs = self.qf(*next_obs_features).max(
            1, keepdim=True
        )[1]
        target_q_values = self.target_qf(*next_obs_features).gather(
            1, best_action_idxs
        )
        y_target = rewards + (1. - terminals) * self.discount * target_q_values
        y_target = y_target.detach()

        # actions is a one-hot vector
        curr_qf = self.qf(*curr_obs_features)
        y_pred = th.sum(curr_qf * actions, dim=1, keepdim=True)
        qf_loss = self.qf_criterion(y_pred, y_target)
        loss += qf_loss

        """
        Update Q networks
        """
        self.qf_optimizer.zero_grad()
        loss.backward()
        self.qf_optimizer.step()

        """
        Soft target network updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf, self.target_qf, self.soft_target_tau
            )

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['Loss'] = np.mean(ptu.get_numpy(loss))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['KL Loss'] = np.mean(ptu.get_numpy(kl_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(y_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(y_target),
            ))

    @property
    def networks(self):
        nets = [
            self.qf,
            self.target_qf,
        ]
        if self.vae is not None:
            nets.append(self.vae)
        return nets

    def get_snapshot(self):
        snapshot = dict(
            qf=self.qf,
            target_qf=self.target_qf,
        )
        if self.vae is not None:
            snapshot['vae'] = self.vae
        return snapshot
