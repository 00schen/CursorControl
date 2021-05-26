import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
from .dqn_trainer import DQNTrainer


class EncDecCQLTrainer(DQNTrainer):
    def __init__(self,
                 rf,
                 qf, target_qf,
                 optimizer,
                 latent_size,
                 encoder=None,
                 temp=1.0,
                 min_q_weight=1.0,
                 add_ood_term=-1,
                 beta=1,
                 sample=True,
                 train_encoder_on_rf=False,
                 **kwargs):
        super().__init__(qf, target_qf, optimizer, **kwargs)
        self.encoder = encoder
        self.rf = rf
        self.temp = temp
        self.min_q_weight = min_q_weight
        self.add_ood_term = add_ood_term
        self.beta = beta
        self.sample = sample
        self.latent_size = latent_size
        self.train_encoder_on_rf = train_encoder_on_rf

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

        if self.encoder is not None:
            curr_latent, curr_kl_loss = self.encoder.sample(curr_goal, eps=eps, return_kl=True)
            next_latent, next_kl_loss = self.encoder.sample(next_goal, eps=eps, return_kl=True)

            # only enforce KL of encoder on next_goal if used to train RF
            if not self.train_encoder_on_rf:
                next_latent = next_latent.detach()
                kl_loss = curr_kl_loss

            else:
                kl_loss = (curr_kl_loss + next_kl_loss) / 2

        else:
            curr_latent, next_latent = curr_goal, next_goal
            kl_loss = 0

        loss += self.beta * kl_loss

        curr_obs_features = [obs, curr_latent]
        next_obs_features = [next_obs, next_latent]

        """
        Rf loss
        """
        num_pos = th.sum(terminals)
        num_neg = terminals.size(0) - num_pos
        pos_weight = None if num_pos == 0 else num_neg / num_pos
        pred_success = self.rf(*next_obs_features)
        rf_loss = th.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred_success, terminals)
        loss += rf_loss
        accuracy = th.eq(terminals, F.sigmoid(pred_success.detach()) > .5).float().mean()

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

        """CQL term"""
        min_qf_loss = th.logsumexp(curr_qf / self.temp, dim=1, ).mean() * self.temp
        min_qf_loss = min_qf_loss - y_pred.mean()

        if self.add_ood_term < 0 or self._n_train_steps_total < self.add_ood_term:
            loss += min_qf_loss * self.min_q_weight

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
            self.eval_statistics['RF Accuracy'] = np.mean(ptu.get_numpy(accuracy))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(y_pred),
            ))

    @property
    def networks(self):
        nets = [
            self.rf,
            self.qf,
            self.target_qf,
        ]
        if self.encoder is not None:
            nets.append(self.encoder)
        return nets

    def get_snapshot(self):
        snapshot = dict(
            rf=self.rf,
            qf=self.qf,
            target_qf=self.target_qf,
        )
        if self.encoder is not None:
            snapshot['encoder'] = self.encoder
        return snapshot
