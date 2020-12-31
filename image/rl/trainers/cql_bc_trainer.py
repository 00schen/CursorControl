import numpy as np
import torch
import torch.optim as optim
import cqlkit.torch.pytorch_util as ptu
from cqlkit.torch.core import np_to_pytorch_batch
from cqlkit.torch.torch_rl_algorithm import TorchTrainer
from cqlkit.core import logger
import time

class CQLBCTrainer:
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
        _,_,_,log_prob,*_ = self.policy(obs,return_log_prob=True)
        bc_loss = -log_prob.mean()
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()
