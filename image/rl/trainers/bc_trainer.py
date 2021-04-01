import torch
import torch.optim as optim
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from .gan_trainer import TorchCycleGANSubTrainer
from collections import OrderedDict


class TorchBCTrainer(TorchTrainer):
    def __init__(
            self,
            policy,
            optimizer
            # policy_lr=1e-3,
            # optimizer_class=optim.Adam,
    ):
        super().__init__()
        self.policy = policy
        self.policy_optimizer = optimizer
        # self.policy_optimizer = optimizer_class(
        # 	self.policy.parameters(),
        # 	lr=policy_lr,
        # )
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        bc_batch = np_to_pytorch_batch(batch)
        obs = bc_batch["observations"]
        actions = bc_batch["actions"]
        dist = self.policy(obs)
        bc_loss = -dist.log_prob(actions).mean()
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()

    def get_diagnostics(self):
        return self.eval_statistics

    @property
    def networks(self):
        return [self.policy]

    def get_snapshot(self):
        return dict(
            policy=self.policy
        )


class VQVAEBCTrainerTorch(TorchBCTrainer):
    def train_from_torch(self, batch):
        obs = batch["observations"]
        actions = batch["actions"]
        labels = torch.argmax(actions, dim=-1)
        vq_loss, pred, _ = self.policy(obs)
        bc_loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss = vq_loss + bc_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()


class DiscreteVAEBCTrainerTorch(TorchBCTrainer):
    def train_from_torch(self, batch):
        obs = batch["observations"]
        actions = batch["actions"]
        labels = torch.argmax(actions, dim=-1)
        eps = torch.normal(mean=torch.zeros((obs.size(0), self.policy.embedding_dim))).to(obs.device)
        kl_loss, pred, sample = self.policy(obs, eps)
        bc_loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss = kl_loss + bc_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

class DiscreteCycleGANBCTrainerTorch(TorchBCTrainer):
    def __init__(
            self,
            policy,
            optimizer,
            encoder,
            gan_kwargs={},
            # policy_lr=1e-3,
            # optimizer_class=optim.Adam,
    ):
        super().__init__(policy,optimizer)
        self.gan_trainer = TorchCycleGANSubTrainer(encoder,**gan_kwargs)
    def train_from_torch(self, batch):
        obs = batch["observations"]
        actions = batch["actions"]
        # gaze = batch['gaze'].flatten()
        labels = torch.argmax(actions, dim=-1)
        pred = self.policy(obs)
        bc_loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss = bc_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        gan_eval_stats = self.gan_trainer.train_from_torch(batch)

        self.eval_statistics.update(gan_eval_stats)

    @property
    def networks(self):
        return [self.policy]+self.gan_trainer.networks

    def get_snapshot(self):
        gan_snapshot = self.gan_trainer.get_snapshot()
        gan_snapshot.update(dict(
            policy=self.policy
        ))
        return gan_snapshot


class DiscreteBCTrainerTorch(TorchBCTrainer):
    def train_from_torch(self, batch):
        obs = batch["observations"]
        actions = batch["actions"]
        # gaze = batch['gaze'].flatten()
        labels = torch.argmax(actions, dim=-1)
        pred = self.policy(obs)
        bc_loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss = bc_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
