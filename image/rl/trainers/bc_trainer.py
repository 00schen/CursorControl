import torch
import torch.optim as optim
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.networks.mlp import Mlp
import rlkit.torch.pytorch_util as ptu
import torch.nn.functional as F
import numpy as np


class TorchBCTrainer(TorchTrainer):
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

    def train_from_torch(self, batch):
        bc_batch = np_to_pytorch_batch(batch)
        obs = bc_batch["observations"]
        actions = bc_batch["actions"]
        dist = self.policy(obs)
        bc_loss = -dist.log_prob(actions).mean()
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()

    @property
    def networks(self):
        return [self.policy]


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
        eps = torch.normal(mean=torch.zeros((obs.size(0), self.policy.embedding_dim))).to(ptu.device)
        kl_loss, pred = self.policy(obs, eps)
        bc_loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss = kl_loss + bc_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()


class DiscreteMixedBCTrainerTorch(TorchBCTrainer):
    def __init__(
            self,
            policy,
            policy_lr=1e-3,
            optimizer_class=optim.Adam,
            discrim_hidden=(32,),
            aux_loss_weight=1,
            l2_weight=0.01,
            reconstruct_weight=1
    ):
        super().__init__(policy=policy, policy_lr=policy_lr, optimizer_class=optimizer_class)
        self.discrim = Mlp(input_size=policy.embedding_dim, output_size=1, hidden_sizes=discrim_hidden,
                           hidden_activation=F.leaky_relu).to(ptu.device, )
        self.discrim_optimizer = optimizer_class(
            self.discrim.parameters(),
            lr=policy_lr,
            betas=(0.5, 0.999)
        )
        self.aux_loss_weight = aux_loss_weight
        self.l2_weight = l2_weight
        self.reconstruct_weight = reconstruct_weight

    def train_from_torch(self, batch):
        obs = batch["observations"]
        actions = batch["actions"]
        labels = torch.argmax(actions, dim=-1)
        batch_size = obs.size(0) // 2
        gaze_obs = obs[:batch_size]
        prior_obs = obs[batch_size:]

        gaze_latent, gaze_pred, reconstruct_loss = self.policy(gaze_obs, gaze=True)
        prior_latent, prior_pred, _ = self.policy(prior_obs, gaze=False)


        pred = torch.cat((gaze_pred, prior_pred), dim=0)
        bc_loss = torch.nn.CrossEntropyLoss()(pred, labels)

        # for i in range(5):
        #     discrim_loss = torch.mean(self.discrim(gaze_latent) - self.discrim(prior_latent))
        #     eps = torch.rand((batch_size, 1)).to(ptu.device)
        #     mix = eps * prior_latent + (1 - eps) * gaze_latent
        #     mix_pred = torch.sum(self.discrim(mix))
        #     mix_grads = torch.autograd.grad(outputs=mix_pred, inputs=mix)[0]
        #     grad_pen = torch.mean((torch.norm(mix_grads, p=2, dim=1) - 1) ** 2)
        #     discrim_loss += 10 * grad_pen
        #
        #     self.discrim_optimizer.zero_grad()
        #     discrim_loss.backward(retain_graph=True)
        #     self.discrim_optimizer.step()

        # fake_gaze = torch.maximum(torch.normal(mean=gaze, std=std), torch.tensor(0).to(ptu.device))
        # fake_latent = self.policy.gaze_encoder(fake_gaze)
        latent = torch.cat((gaze_latent, prior_latent), dim=0)
        discrim_pred = self.discrim(latent)
        discrim_labels = torch.cat((torch.zeros((gaze_latent.size(0), 1)),
                                    0.9 * torch.ones((prior_latent.size(0), 1))), dim=0).to(ptu.device)
        pos_weight = torch.tensor(gaze_latent.size(0) / prior_latent.size(0)).to(ptu.device)
        discrim_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(discrim_pred, discrim_labels)

        self.discrim_optimizer.zero_grad()
        discrim_loss.backward(retain_graph=True)
        self.discrim_optimizer.step()

        # aux_pred = self.discrim(gaze_latent)
        # aux_loss = -torch.mean(aux_pred)

        # fake_aux = torch.maximum(torch.normal(mean=gaze, std=std), torch.tensor(0).to(ptu.device))
        # fake_aux_latent = self.policy.gaze_encoder(fake_aux)
        aux_pred = self.discrim(gaze_latent)
        aux_labels = torch.zeros((gaze_latent.size(0), 1)).to(ptu.device)
        aux_loss = -torch.nn.BCEWithLogitsLoss()(aux_pred, aux_labels)

        l2_reg = 0
        for encoder in self.policy.gaze_encoders:
            for param in encoder.parameters():
                l2_reg += torch.norm(param)

        loss = bc_loss + (self.aux_loss_weight * aux_loss) + (self.l2_weight * l2_reg) + \
               self.reconstruct_weight + reconstruct_loss

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()


class DiscreteBCTrainerTorch(TorchBCTrainer):
    def train_from_torch(self, batch):
        obs = batch["observations"]
        actions = batch["actions"]
        labels = torch.argmax(actions, dim=-1)
        pred = self.policy(obs)
        bc_loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss = bc_loss
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
