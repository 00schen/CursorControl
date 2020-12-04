import numpy as np
import torch
import torch.optim as optim
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core import logger
import time

class BCTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,

            policy_lr=1e-3,
            policy_weight_decay=0,
            optimizer_class=optim.Adam,

            bc_num_pretrain_steps=0,
            bc_batch_size=128,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            weight_decay=policy_weight_decay,
            lr=policy_lr,
        )
        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.bc_batch_size = bc_batch_size

    def get_batch_from_buffer(self, replay_buffer, batch_size):
        batch = replay_buffer.random_batch(batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch

    def run_bc_batch(self, replay_buffer, policy):
        batch = self.get_batch_from_buffer(replay_buffer, self.bc_batch_size)
        o = batch["observations"]
        u = batch["actions"]
        og = o
        dist = policy(og)
        pred_u, log_pi = dist.rsample_and_logprob()
        stats = dist.get_diagnostics()

        mse = (pred_u - u) ** 2
        mse_loss = mse.mean()

        policy_logpp = dist.log_prob(u, )
        logp_loss = -policy_logpp.mean()
        policy_loss = logp_loss

        return policy_loss, logp_loss, mse_loss, stats

    def pretrain_policy_with_bc(self, train_buffer):
        optimizer = self.policy_optimizer
        prev_time = time.time()
        for i in range(self.bc_num_pretrain_steps):
            train_policy_loss, train_logp_loss, train_mse_loss, train_stats = self.run_bc_batch(train_buffer, policy)
            train_policy_loss = train_policy_loss

            optimizer.zero_grad()
            train_policy_loss.backward()
            optimizer.step()

            if i % 100==0:
                stats = {
                "pretrain_bc/batch": i,
                "pretrain_bc/Train Logprob Loss": ptu.get_numpy(train_logp_loss),
                "pretrain_bc/Train MSE": ptu.get_numpy(train_mse_loss),
                "pretrain_bc/train_policy_loss": ptu.get_numpy(train_policy_loss),
                "pretrain_bc/epoch_time":time.time()-prev_time,
                }
                logger.record_dict(stats)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)
                prev_time = time.time()

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.policy,
        ]
        return nets

    def get_snapshot(self):
        return dict(
            policy=self.policy,
        )
