import torch as th
import numpy as np
from collections import OrderedDict

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class EncDecSACTrainer(TorchTrainer):
    def __init__(self,
                 vae,
                 prev_vae,
                 policy,
                 optimizer,
                 latent_size,
                 temp=1.0,
                 beta=1,
                 sample=True,
                 use_supervised='none',
                 ):
        super().__init__()
        self.policy = policy
        self.optimizer = optimizer
        self.vae = vae
        self.prev_vae = prev_vae
        self.temp = temp
        self.beta = beta
        self.sample = sample
        self.latent_size = latent_size
        self.use_supervised = use_supervised

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
        latents = batch['curr_latents']
        goals = batch['curr_goal']
        actions = batch['actions']

        loss = ptu.zeros(1)

        eps = th.normal(ptu.zeros((inputs.size(0), self.latent_size)), 1) if self.sample else None
        pred_latent, kl_loss = self.vae.sample(inputs, eps=eps, return_kl=True)
        success_indices = episode_success.flatten() == 1

        if self.prev_vae is not None:
            target_latent = self.prev_vae.sample(goals, eps=None)
        else:
            target_latent = goals

        latent_error = th.linalg.norm(pred_latent - target_latent, dim=-1)

        # self.use_supervised unused for now
        target_mean = self.policy(obs, target_latent).mean.detach()
        pred_mean = self.policy(obs, pred_latent).mean
        supervised_loss = th.nn.MSELoss()(pred_mean, target_mean)

        loss += supervised_loss + self.beta * kl_loss

        """
        Update Q networks
        """
        self.optimizer.zero_grad()
        loss.backward()
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

    @property
    def networks(self):
        nets = [
            self.vae,
            self.policy
        ]
        return nets

    def get_snapshot(self):
        return dict(
            vae=self.vae,
            policy=self.policy
        )
