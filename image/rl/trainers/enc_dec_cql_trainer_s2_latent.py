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
                 encoder,
                 recon_decoder,
                 prev_encoder,
                 qf, target_qf,
                 optimizer,
                 latent_size,
                 temp=1.0,
                 min_q_weight=1.0,
                 add_ood_term=-1,
                 beta=1,
                 sample=True,
                 prev_sample=True,
                 use_supervised='none',
                 pos_weight=1,
                 **kwargs):
        super().__init__(qf, target_qf, optimizer, **kwargs)
        # self.qf_optimizer also optimizes encoder
        self.rf = rf
        self.encoder = encoder
        self.recon_decoder = recon_decoder
        self.prev_encoder = prev_encoder
        self.temp = temp
        self.min_q_weight = min_q_weight
        self.add_ood_term = add_ood_term
        self.beta = beta
        self.sample = sample
        self.prev_sample = prev_sample
        self.latent_size = latent_size
        self.use_supervised = use_supervised
        self.pos_weight = pos_weight

    def train_from_torch(self, batch):
        episode_success = batch['episode_success']
        obs = batch['observations']
        inputs = th.cat((batch['curr_gaze_features'], obs), dim=1)
        latents = batch['curr_latents']
        goals = batch['curr_goal']
        actions = batch['actions']

        loss = ptu.zeros(1)

        eps = th.normal(ptu.zeros((inputs.size(0), self.latent_size)), 1) if self.sample else None
        pred_latent, kl_loss = self.encoder.sample(inputs, eps=eps, return_kl=True)
        success_indices = episode_success.flatten() == 1

        if 'useneg' in self.use_supervised:
            weight = th.where(success_indices, self.pos_weight, 1)
            surrogate_probs = th.exp(-(th.norm(pred_latent - latents, dim=-1) ** 2))
            supervised_loss = th.nn.BCELoss(weight=weight)(surrogate_probs, success_indices)
        else:
            if self.prev_encoder is not None:
                target_latent = self.prev_encoder.sample(goals, eps=None)
            else:
                target_latent = goals
            if 'kl' in self.use_supervised:
                target_q_dist = th.log_softmax(self.qf(obs, target_latent) * self.temp, dim=1).detach()
                pred_q_dist = th.log_softmax(self.qf(obs, pred_latent) * self.temp,
                                             dim=1)
                supervised_loss = th.nn.KLDivLoss(log_target=True)(pred_q_dist, target_q_dist)
            elif 'AWR' in self.use_supervised:
                pred_mean, pred_logvar = self.encoder.encode(inputs)
                kl_loss = self.encoder.kl_loss(pred_mean, pred_logvar)
                supervised_loss = th.nn.GaussianNLLLoss(reduction='none')(pred_mean, latents, th.exp(pred_logvar))
                weights = th.where(success_indices, 1., 0.)
                supervised_loss = th.mean(supervised_loss * weights)
            else:
                supervised_loss = th.nn.MSELoss()(pred_latent[success_indices], latents[success_indices])

        loss += supervised_loss + self.beta * kl_loss

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
            self.eval_statistics['SL Loss'] = np.mean(ptu.get_numpy(supervised_loss))
            self.eval_statistics['KL Loss'] = np.mean(ptu.get_numpy(kl_loss))


    @property
    def networks(self):
        nets = [
            self.rf,
            self.encoder,
            self.recon_decoder,
            self.qf,
            self.target_qf,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            rf=self.rf,
            encoder=self.encoder,
            recon_decoder=self.recon_decoder,
            qf=self.qf,
            target_qf=self.target_qf,
        )
