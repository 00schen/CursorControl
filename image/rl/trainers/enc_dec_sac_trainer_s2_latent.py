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
                 prev_incl_state=False,
                 window_size=None
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
        self.window_size = window_size

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        vae = self.vaes[self._num_train_steps % len(self.vaes)]

        obs = batch['observations']

        feature_name = lambda x: 'curr_' + x if self.window_size is None else x + '_hist'
        features = th.cat([batch[feature_name(key)] for key in self.feature_keys], dim=1)
        latents = batch['curr_latents']
        goals = batch['curr_goal']
        curr_goal_set = batch.get('curr_goal_set')

        has_goal_set = curr_goal_set is not None
        batch_size = obs.shape[0]


        encoder_features = [features]
        if self.incl_state:
            encoder_obs = obs if self.window_size is None else batch['obs_hist']
            encoder_features.append(encoder_obs)

            # goal set and window does not work together
            if has_goal_set:
                curr_goal_set_flat = curr_goal_set.reshape((batch_size, -1))
                encoder_features.append(curr_goal_set_flat)

        mean, logvar = vae.encode(th.cat(encoder_features, dim=-1))
        if self.window_size is not None:
            if self.objective == 'goal':
                mask = th.unsqueeze(batch['hist_mask'], -1)
                mean = th.sum(mean * mask, dim=1) / th.sum(mask, dim=1)
                sigma_squared = None
            else:
                mean, sigma_squared = self._product_of_gaussians(mean, logvar, batch['hist_mask'])
        else:
            sigma_squared = th.exp(logvar)

        # regress directly to goals
        if self.objective == 'goal':
            supervised_loss = th.nn.MSELoss()(mean, goals)
            kl_loss = ptu.zeros(1)
            latent_error = ptu.zeros(1)

        else:
            kl_loss = vae.kl_loss(mean, th.log(sigma_squared))
            pred_latent = mean
            if self.sample:
                pred_latent = pred_latent + th.sqrt(sigma_squared) * ptu.normal(th.zeros(pred_latent.shape), 1)

            if self.prev_vae is not None:
                prev_encoder_features = [goals]
                if self.prev_incl_state:
                    prev_encoder_features.append(obs)
                    if has_goal_set:
                        curr_goal_set_flat = curr_goal_set.reshape((batch_size, -1))
                        prev_encoder_features.append(curr_goal_set_flat)

                target_latent = self.prev_vae.sample(th.cat(prev_encoder_features, dim=-1), eps=None)
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

        loss = supervised_loss + self.beta * kl_loss

        """
        Update Q networks
        """
        self.optimizer.zero_grad()
        loss.backward()
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

    def _product_of_gaussians(self, means, logvars, mask):
        sigmas_squared = th.clamp(th.exp(logvars), min=1e-7)
        mask = th.unsqueeze(mask, -1)
        sigma_squared = 1. / th.sum(th.reciprocal(sigmas_squared) * mask, dim=1)
        mean = sigma_squared * th.sum((means / sigmas_squared) * mask, dim=1)
        return mean, sigma_squared

    def compute_kl_div(self, mean, sigma_squared):
        prior = th.distributions.Normal(ptu.zeros(self.latent_size), ptu.ones(self.latent_size))
        posteriors = [th.distributions.Normal(m, th.sqrt(s)) for m, s in zip(th.unbind(mean), th.unbind(sigma_squared))]
        kl_divs = [th.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        return th.mean(th.sum(th.stack(kl_divs), dim=-1))

    @property
    def networks(self):
        nets = self.vaes + [self.policy]
        return nets

    def get_snapshot(self):
        return dict(
            vaes=tuple(self.vaes),
            policy=self.policy
        )
