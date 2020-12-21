import numpy as np
import torch
import torch.optim as optim
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core import logger
import time

class BCTrainer:
    def __init__(
            self,
            policy,
            policy_lr=1e-3,
            optimizer_class=optim.Adam,
    ):
        super().__init__()
        self.policy = policy
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

    def pretrain(self, bc_batch):
        bc_batch = np_to_pytorch_batch(bc_batch)
        obs = bc_batch["observations"]
        actions = bc_batch["actions"]
        dist = self.policy(obs)
        bc_loss = -dist.log_prob(actions).mean()
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()
