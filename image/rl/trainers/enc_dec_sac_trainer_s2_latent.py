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
                 qf1,
                 qf2,
                 optimizer,
                 latent_size,
                 feature_keys,
                 beta=1,
                 sample=True,
                 objective='kl',
                 grad_norm_clip=1,
                 ):
        super().__init__()
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.optimizer = optimizer
        self.vae = vae
        self.prev_vae = prev_vae
        self.beta = beta
        self.sample = sample
        self.latent_size = latent_size
        self.feature_keys = feature_keys
        self.objective = objective
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
        features = th.cat([batch['curr_' + key] for key in self.feature_keys], dim=1)
        latents = batch['curr_latents']
        goals = batch['curr_goal']
        curr_goal_set = batch.get('curr_goal_set')

        has_goal_set = curr_goal_set is not None
        batch_size = obs.shape[0]

        loss = ptu.zeros(1)

        if has_goal_set:
            curr_goal_set_flat = curr_goal_set.reshape((batch_size, -1))
            encoder_features = [features, obs, curr_goal_set_flat]
        else:
            encoder_features = [features, obs]

        eps = th.normal(ptu.zeros((batch_size, self.latent_size)), 1) if self.sample else None
        pred_latent, kl_loss = self.vae.sample(th.cat(encoder_features, dim=1), eps=eps, return_kl=True)

        if self.prev_vae is not None:
            if self.prev_vae.encoder.input_size == goals.shape[-1]:
                target_latent = self.prev_vae.sample(goals, eps=None)
            else:
                target_latent = self.prev_vae.sample(th.cat((goals,obs), dim=1), eps=None)
        else:
            target_latent = goals

        latent_error = th.linalg.norm(pred_latent - target_latent, dim=-1)

        if has_goal_set:
            curr_goal_set_flat = curr_goal_set.reshape((batch_size, -1))
            target_policy_features = [obs, curr_goal_set_flat, target_latent]
            pred_policy_features = [obs, curr_goal_set_flat, pred_latent]
        else:
            target_policy_features = [obs, target_latent]
            pred_policy_features = [obs, pred_latent]

        if self.objective == 'kl':
            target_mean = self.policy(*target_policy_features).mean.detach()
            pred_mean = self.policy(*pred_policy_features).mean
            supervised_loss = th.mean(th.sum(th.nn.MSELoss(reduction='none')(pred_mean, target_mean), dim=-1))
        elif self.objective == 'awr':
            pred_mean, pred_logvar = self.vae.encode(th.cat(encoder_features, dim=1))
            kl_loss = self.vae.kl_loss(pred_mean, pred_logvar)
            supervised_loss = th.nn.GaussianNLLLoss()(pred_mean, latents.detach(), th.exp(pred_logvar))
        elif self.objective == 'joint':
            dist = self.policy(*pred_policy_features)

            new_obs_actions, log_pi = dist.rsample_and_logprob()
            log_pi = log_pi.unsqueeze(-1)

            if has_goal_set:
                new_qf_features = [obs, curr_goal_set_flat, goals, new_obs_actions]
            else:
                new_qf_features = [obs, goals, new_obs_actions]

            q_new_actions = th.min(
                self.qf1(*new_qf_features),
                self.qf2(*new_qf_features),
            )

            supervised_loss = (-q_new_actions).mean()
        else:
            raise NotImplementedError()

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
        if self.grad_norm_clip is not None:
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
            self.policy
        ]
        return nets

    def get_snapshot(self):
        return dict(
            vae=self.vae,
            policy=self.policy
        )