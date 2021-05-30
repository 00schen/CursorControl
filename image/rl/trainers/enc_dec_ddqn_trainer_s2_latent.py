import torch as th
import numpy as np
from collections import OrderedDict

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class EncDecDQNTrainer(TorchTrainer):
    def __init__(self,
                 vae,
                 prev_vae,
                 qf, target_qf,
                 optimizer,
                 latent_size,
                 temp=1.0,
                 beta=1,
                 sample=True,
                 use_supervised='none',
                 grad_norm_clip=0.5
                 ):

        super().__init__()
        self.qf = qf
        self.target_qf = target_qf
        self.optimizer = optimizer
        self.vae = vae
        self.prev_vae = prev_vae
        self.temp = temp
        self.beta = beta
        self.sample = sample
        self.latent_size = latent_size
        self.use_supervised = use_supervised
        self.grad_norm_clip = grad_norm_clip

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        episode_success = batch['episode_success']
        obs = batch['observations']
        inputs = th.cat((batch['curr_gaze_features'], obs), dim=1)
        gazes = batch['curr_gaze_features']
        latents = batch['curr_latents']
        goals = batch['curr_goal']

        loss = ptu.zeros(1)

        eps = th.normal(ptu.zeros((inputs.size(0), self.latent_size)), 1) if self.sample else None
        pred_latent, kl_loss = self.vae.sample(th.cat((gazes, obs), dim=1), eps=eps, return_kl=True)
        success_indices = episode_success.flatten() == 1

        if self.prev_vae is not None:
            target_latent = self.prev_vae.sample(th.cat((goals, obs), dim=1), eps=None)
        else:
            target_latent = goals

        latent_error = th.linalg.norm(pred_latent - target_latent, dim=-1)

        if 'kl' in self.use_supervised:
            target_q_dist = th.log_softmax(self.qf(obs, target_latent) * self.temp, dim=1).detach()
            pred_q_dist = th.log_softmax(self.qf(obs, pred_latent) * self.temp, dim=1)
            supervised_loss = th.nn.KLDivLoss(log_target=True, reduction='batchmean')(pred_q_dist, target_q_dist)
        elif 'AWR' in self.use_supervised:
            pred_mean, pred_logvar = self.vae.encode(inputs)
            kl_loss = self.vae.kl_loss(pred_mean, pred_logvar)
            supervised_loss = th.nn.GaussianNLLLoss(reduction='none')(pred_mean, latents, th.exp(pred_logvar))
            weights = th.where(success_indices, 1., 0.)
            supervised_loss = th.mean(supervised_loss * weights)
        else:
            supervised_loss = th.mean(th.sum(th.nn.MSELoss(reduction='none')(pred_latent, target_latent.detach()),
                                             dim=-1))

        loss += supervised_loss + self.beta * kl_loss

        """
        Update Q networks
        """
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for p in self.vae.encoder.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        th.nn.utils.clip_grad_norm_(self.vae.encoder.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        """
        Save some statistics for eval using just one batch.
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['Loss'] = np.mean(ptu.get_numpy(loss))
            self.eval_statistics['SL Loss'] = np.mean(ptu.get_numpy(supervised_loss))
            self.eval_statistics['KL Loss'] = np.mean(ptu.get_numpy(kl_loss))
            self.eval_statistics['Latent Error'] = np.mean(ptu.get_numpy(latent_error))
            self.eval_statistics['Gradient Norm'] = np.mean(total_norm)


    @property
    def networks(self):
        nets = [
            self.vae,
            self.qf,
            self.target_qf,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            vae=self.vae,
            qf=self.qf,
            target_qf=self.target_qf,
        )
