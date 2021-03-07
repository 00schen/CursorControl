import torch as th
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
import rlkit.torch.pytorch_util as ptu
from rl.trainers.ddqn_cql_trainer import DDQNCQLTrainer




class QRDDQNCQLTrainer(DDQNCQLTrainer):
    def __init__(self, qf, target_qf,
                 adam_eps=0.0003125,
                 kappa=1.0,
                 **kwargs):
        super().__init__(qf, target_qf, **kwargs)
        self.adam_eps = adam_eps
        self.kappa = kappa
        self.qf_optimizer = optim.Adam(
            self.qf.parameters(),
            lr=self.learning_rate,
            eps=self.adam_eps
        )

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
                rf_obs, rf_next_obs, rf_noop = self.mixup(obs, next_obs, noop)
                pred_reward = self.rf(rf_obs, rf_next_obs)
                rf_loss = F.binary_cross_entropy_with_logits(pred_reward, rf_noop,)

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

        _, target_qvalues = self.qf(next_obs)
        best_action_idxs = th.argmax(target_qvalues, dim=1)
        target_logits, _ = self.target_qf(next_obs)
        targets = th.stack([target_logits[i].index_select(0, best_action_idxs[i]) for i in range(batch_size)]).squeeze(1)
        y_target = rewards + discount_with_terminal * targets
        y_target = y_target[:, None, :].detach()

        # actions is a one-hot vector
        curr_logits, curr_qvalues = self.qf(obs)
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
        loss = th.mean(th.sum(th.mean(quantile_huber_loss, dim=2), dim=1))

        """CQL term"""
        min_qf_loss = th.logsumexp(curr_qvalues / self.temp, dim=1).mean() * self.temp
        min_qf_loss = min_qf_loss - curr_chosen_qvalues.mean()
        if self.add_ood_term < 0 or self._n_train_steps_total < self.add_ood_term:
            loss += min_qf_loss * self.min_q_weight
            # loss -= curr_chosen_qvalues[batch_size:].mean()
        # prob = th.nn.LogSoftmax(dim=-1)(curr_qvalues)
        # loss = -th.mean(prob * batch['actions'])
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
