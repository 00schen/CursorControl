from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch
import torch.optim as optim
import numpy as np
from collections import OrderedDict


class RewardTrainer(TorchTrainer):
    def __init__(
            self,
            rew_net,
            learning_rate,
            mixup=1
    ):
        super().__init__()
        self.rew_net = rew_net
        self.learning_rate = learning_rate
        self.mixup = mixup

        self.optimizer = optim.Adam(
            self.rew_net.parameters(),
            lr=self.learning_rate
        )
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        rewards = batch['rewards'] + 1
        obs = batch['observations']
        next_obs = batch['next_observations']
        # actions = batch['actions']

        """
        Compute loss
        """
        if self.mixup > 0:
            assert len(obs) % 2 == 0
            batch_size = len(obs) // 2
            mix_weights = torch.distributions.beta.Beta(self.mixup, self.mixup).sample((batch_size, 1))
            next_obs = mix_weights * next_obs[:batch_size] + (1 - mix_weights) * next_obs[batch_size:]
            #action_choices = mix_weights.repeat(1, actions.size()[-1]) >= 0.5
            #actions = torch.where(action_choices, actions[:batch_size], actions[batch_size:])
            rewards = mix_weights * rewards[:batch_size] + (1 - mix_weights) * rewards[batch_size:]

        self.rew_net.train(True)
        # pred = torch.sum(self.rew_net(obs) * actions, dim=1, keepdim=True)
        pred = self.rew_net(next_obs)
        loss = torch.nn.BCELoss()(pred, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.rew_net.train(False)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['train_loss'] = loss.item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    def eval_new_paths(self, paths):
        p = {}
        for k in paths[0].keys():
            p[k] = np.concatenate(list(path[k] for path in paths))
        # valid_pred = np.sum(self.rew_net(torch.from_numpy(
        #     self.valid_data['observations'].astype(np.float32))).detach().numpy() * self.valid_data['actions'],
        #               axis=1)
        pred = self.rew_net(torch.from_numpy(
            p['next_observations'].astype(np.float32))).detach().numpy().flatten()
        labels = p['rewards'].flatten() + 1
        correct = (pred >= 0.5) == (labels >= 0.5)
        pos_correct = np.mean(correct[labels >= 0.5])
        neg_correct = np.mean(correct[labels < 0.5])
        loss = -np.mean(np.concatenate((np.log(pred[labels >= 0.5]), np.log(1 - pred[labels < 0.5]))))
        self.eval_statistics['pos_correct'] = pos_correct
        self.eval_statistics['neg_correct'] = neg_correct
        self.eval_statistics['valid_loss'] = loss

    @property
    def networks(self):
        return [
            self.rew_net
        ]
