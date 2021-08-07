import torch as th
import numpy as np
from collections import OrderedDict

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class EncDecSACTrainer(TorchTrainer):
    def __init__(self,
                 vaes,
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
                 incl_state=True,
                 prev_incl_state=False
                 ):
        super().__init__()
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.optimizer = optimizer
        self.vaes = vaes
        self.prev_vae = prev_vae
        self.beta = beta
        self.sample = sample
        self.latent_size = latent_size
        self.feature_keys = feature_keys
        self.objective = objective
        self.grad_norm_clip = grad_norm_clip
        self.incl_state = incl_state
        self.prev_incl_state = prev_incl_state

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        vae = self.vaes[self._num_train_steps % len(self.vaes)]

        episode_success = batch['episode_success']
        obs = batch['observations']
        features = th.cat([batch['curr_' + key] for key in self.feature_keys], dim=1)
        latents = batch['curr_latents']
        goals = batch['curr_goal']
        batch_size = obs.shape[0]
        loss = ptu.zeros(1)

        encoder_features = [features]
        if self.incl_state:
            encoder_features.append(obs)

        eps = th.normal(ptu.zeros((batch_size, self.latent_size)), 1) if self.sample else None
        pred_latent, kl_loss = vae.sample(th.cat(encoder_features, dim=1), eps=eps, return_kl=True)

        if self.prev_vae is not None:
            prev_encoder_features = [goals]
            if self.prev_incl_state:
                prev_encoder_features.append(obs)

            target_latent = self.prev_vae.sample(th.cat(prev_encoder_features, dim=-1), eps=None)
        else:
            target_latent = goals

        latent_error = th.linalg.norm(pred_latent - target_latent, dim=-1)

        target_policy_features = [obs, target_latent]
        pred_policy_features = [obs, pred_latent]

        if self.objective == 'kl':
            target_mean = self.policy(*target_policy_features).mean.detach()
            pred_mean = self.policy(*pred_policy_features).mean
            supervised_loss = th.mean(th.sum(th.nn.MSELoss(reduction='none')(pred_mean, target_mean), dim=-1))
        elif self.objective == 'normal_kl':
            target = self.policy(*target_policy_features).normal
            pred = self.policy(*pred_policy_features).normal
            supervised_loss = th.mean(th.distributions.kl.kl_divergence(target, pred))
        elif self.objective == 'awr':
            pred_mean, pred_logvar = vae.encode(th.cat(encoder_features, dim=1))
            kl_loss = vae.kl_loss(pred_mean, pred_logvar)
            supervised_loss = th.nn.GaussianNLLLoss()(pred_mean, latents.detach(), th.exp(pred_logvar))
        elif self.objective == 'latent':
            supervised_loss = th.nn.MSELoss()(pred_latent, target_latent.detach())
        elif self.objective == 'joint':
            dist = self.policy(*pred_policy_features)

            new_obs_actions, log_pi = dist.rsample_and_logprob()
            log_pi = log_pi.unsqueeze(-1)

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
        for p in vae.encoder.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if self.grad_norm_clip is not None:
            th.nn.utils.clip_grad_norm_(vae.encoder.parameters(), self.grad_norm_clip)
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
        nets = self.vaes + [self.policy]
        return nets

    def get_snapshot(self):
        return dict(
            vaes=tuple(self.vaes),
            policy=self.policy
        )
