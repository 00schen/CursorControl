from rl.trainers.ddqn_cql_trainer import DDQNCQLTrainer
import torch as th
import rlkit.torch.pytorch_util as ptu
import torch.nn.functional as F
import numpy as np
from rlkit.core.eval_util import create_stats_ordered_dict
import torch.optim as optim
from rlkit.torch.networks.mlp import Mlp



class QRDDQNCQLTrainer(DDQNCQLTrainer):
    def __init__(self, qf, target_qf,
                 adam_eps=0.0003125,
                 kappa=1.0,
                 discrim_hidden=(32,),
                 aux_loss_weight=1,
                 num_discrims=3,
                 l2_weight=0.01,
                 reconstruct_weight=1,
                 **kwargs):
        super().__init__(qf, target_qf, **kwargs)
        self.adam_eps = adam_eps
        self.kappa = kappa
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
            eps=self.adam_eps
        )

        # self.discrim = Mlp(input_size=qf.embedding_dim, output_size=1, hidden_sizes=discrim_hidden,
        #                    hidden_activation=F.leaky_relu).to(ptu.device)

        self.num_discrims = num_discrims
        self.discrims = []
        self.discrim_optimizers = []
        for i in range(num_discrims):
            discrim = Mlp(input_size=qf.embedding_dim, output_size=1, hidden_sizes=discrim_hidden,
                          hidden_activation=F.leaky_relu).to(ptu.device)
            self.discrims.append(discrim)
            self.discrim_optimizers.append(optim.Adam(discrim.parameters(), lr=1e-4, betas=(0.5, 0.999)))

        self.aux_loss_weight = aux_loss_weight
        self.l2_weight = l2_weight
        self.reconstruct_weight = reconstruct_weight

    def eval_qf_mixed_batch(self, qf, obs, train=True):
        batch_size = obs.size(0) // 2
        gaze_obs = obs[:batch_size]
        prior_obs = obs[batch_size:]
        gaze_latent, reconstruct_loss, gaze_logits, gaze_qvalues = qf(gaze_obs, gaze=True, train=train)
        prior_latent, _, prior_logits, prior_qvalues = qf(prior_obs, gaze=False, train=train)
        logits = th.cat((gaze_logits, prior_logits), dim=0)
        qvalues = th.cat((gaze_qvalues, prior_qvalues), dim=0)
        return gaze_latent, prior_latent, logits, qvalues, reconstruct_loss


    def train_from_torch(self, batch):
        noop = th.clamp(batch['rewards'] + 1, 0, 1)
        terminals = batch['terminals']
        actions = batch['actions']
        obs = batch['observations']
        next_obs = batch['next_observations']

        """
        Reward and R loss
        """
        if not self.ground_truth:
            rewards = (1 - self.rf(obs, next_obs).exp()).log() * -1 * batch['rewards']
            if self._n_train_steps_total % self.reward_update_period == 0:
                noop_prop = noop.mean().item()
                noop_prop = max(1e-4, 1 - noop_prop) / max(1e-4, noop_prop)
                rf_obs, rf_next_obs, rf_noop = self.mixup(obs, next_obs, noop)
                pred_reward = self.rf(rf_obs, rf_next_obs)
                rf_loss = F.binary_cross_entropy_with_logits(pred_reward, rf_noop, pos_weight=ptu.tensor([noop_prop]))

                self.rf_optimizer.zero_grad()
                rf_loss.backward()
                self.rf_optimizer.step()
        else:
            rewards = batch['rewards']

        """
        Q loss
        """
        batch_size = rewards.size(0)
        is_terminal_multipler = 1. - terminals
        discount_with_terminal = self.discount * is_terminal_multipler

        _, _, _, target_qvalues, _ = self.eval_qf_mixed_batch(self.qf, next_obs, train=False)
        best_action_idxs = th.argmax(target_qvalues, dim=1)
        _, _, target_logits, _, _ = self.eval_qf_mixed_batch(self.target_qf, next_obs, train=False)
        targets = th.stack([target_logits[i].index_select(0, best_action_idxs[i]) for i in range(batch_size)]).squeeze(1)
        y_target = rewards + discount_with_terminal * targets
        y_target = y_target[:, None, :].detach()

        # actions is a one-hot vector
        gaze_latent, prior_latent, curr_logits, curr_qvalues, reconstruct_loss = self.eval_qf_mixed_batch(self.qf, obs)
        curr_chosen_qvalues = th.sum(curr_qvalues * actions, dim=1)
        n_atoms = curr_logits.size(-1)
        actions = actions[:, :, None].repeat((1, 1, n_atoms))
        y_pred = th.sum(curr_logits * actions, dim=1)[:, :, None]

        bellman_errors = y_target - y_pred
        huber_loss = (th.abs(bellman_errors) <= self.kappa).double() * 0.5 * bellman_errors ** 2 + \
                     (th.abs(bellman_errors) > self.kappa).double() * self.kappa * (th.abs(bellman_errors)
                                                                                    - 0.5 * self.kappa)
        tau_hat = (th.arange(n_atoms).to(huber_loss.device) + 0.5) / n_atoms
        quantile_huber_loss = th.abs(tau_hat[None, :, None] - (bellman_errors < 0).double()) * huber_loss
        rl_loss = loss = th.mean(th.sum(th.mean(quantile_huber_loss, dim=2), dim=1))

        """CQL term"""
        min_qf_loss = th.logsumexp(curr_qvalues / self.temp, dim=1).mean() * self.temp
        min_qf_loss = min_qf_loss - curr_chosen_qvalues.mean()
        if self.add_ood_term < 0 or self._n_train_steps_total < self.add_ood_term:
            loss += min_qf_loss * self.min_q_weight
            # loss -= curr_chosen_qvalues[batch_size:].mean()
        # prob = th.nn.LogSoftmax(dim=-1)(curr_qvalues)
        # loss = -th.mean(prob * batch['actions'])

        index = np.random.randint(self.num_discrims)
        discrim = self.discrims[index]
        discrim_optimizer = self.discrim_optimizers[index]

        latent = th.cat((gaze_latent, prior_latent), dim=0)
        discrim_pred = discrim(latent)
        discrim_labels = th.cat((th.zeros((gaze_latent.size(0), 1)),
                                0.9 * th.ones((prior_latent.size(0), 1))), dim=0).to(ptu.device)
        pos_weight = th.tensor(gaze_latent.size(0) / prior_latent.size(0)).to(ptu.device)
        discrim_loss = th.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(discrim_pred, discrim_labels)

        discrim_optimizer.zero_grad()
        discrim_loss.backward(retain_graph=True)
        discrim_optimizer.step()

        # for i in range(5):
        #     discrim_loss = th.mean(self.discrim(gaze_latent) - self.discrim(prior_latent))
        #     eps = th.rand((batch_size // 2, 1)).to(ptu.device)
        #     mix = eps * prior_latent + (1 - eps) * gaze_latent
        #     mix_pred = th.sum(self.discrim(mix))
        #     mix_grads = th.autograd.grad(outputs=mix_pred, inputs=mix)[0]
        #     grad_pen = th.mean((th.norm(mix_grads, p=2, dim=1) - 1) ** 2)
        #     discrim_loss += 10 * grad_pen
        #
        #     self.discrim_optimizer.zero_grad()
        #     discrim_loss.backward(retain_graph=True)
        #     self.discrim_optimizer.step()

        aux_pred = discrim(gaze_latent)
        aux_labels = th.ones((gaze_latent.size(0), 1)).to(ptu.device)
        aux_loss = th.nn.BCEWithLogitsLoss()(aux_pred, aux_labels)
        loss += self.aux_loss_weight * aux_loss

        l2_reg = 0
        for encoder in self.qf.gaze_encoders:
            for param in encoder.parameters():
                l2_reg += th.norm(param)
        loss += self.l2_weight * l2_reg
        loss += self.reconstruct_weight + reconstruct_loss

        # aux_pred = self.discrim(gaze_latent)
        # aux_loss = -th.mean(aux_pred)
        # loss += self.aux_loss_weight * aux_loss
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
            # self.eval_statistics['RF Loss'] = np.mean(ptu.get_numpy(rf_loss))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(loss))
            self.eval_statistics['QF OOD Loss'] = np.mean(ptu.get_numpy(min_qf_loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'R Predictions',
                ptu.get_numpy(rewards),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(y_pred),
            ))
