from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch
import torch.optim as optim


class RewardTrainer(TorchTrainer):
    def __init__(
            self,
            rew_net,
            learning_rate=1e-3,
    ):
        super().__init__()
        self.rew_net = rew_net
        self.learning_rate = learning_rate

        self.optimizer = optim.Adam(
            self.rew_net.parameters(),
            lr=self.learning_rate
        )

        self._n_train_steps_total = 0

    def train_from_torch(self, batch):
        rewards = batch['rewards'] + 1
        obs = batch['observations']
        actions = batch['actions']

        """
        Compute loss
        """

        pred = torch.sum(self.rew_net(obs) * actions, dim=1, keepdim=True)
        loss = torch.nn.BCELoss()(pred, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._n_train_steps_total += 1

    @property
    def networks(self):
        return [
            self.rew_net
        ]
