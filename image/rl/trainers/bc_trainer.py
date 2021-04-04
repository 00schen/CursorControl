import torch
import torch.optim as optim
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.networks.mlp import Mlp
import rlkit.torch.pytorch_util as ptu
import torch.nn.functional as F
import numpy as np
import itertools
import h5py


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
        self.discrim = Mlp(input_size=policy.shared_dim, output_size=1, hidden_sizes=discrim_hidden,
                           ).to(ptu.device)
        self.discrim_optimizer = optimizer_class(
            itertools.chain(self.discrim.parameters(), self.policy.shared_decoder.parameters()),
            lr=policy_lr,
            betas=(0.5, 0.999)
        )
        self.aux_loss_weight = aux_loss_weight
        self.l2_weight = l2_weight
        self.reconstruct_weight = reconstruct_weight

        data_paths = ['image/rl/gaze_capture/gaze_data_train.h5']
        gaze_data = []
        for path in data_paths:
            loaded = h5py.File(path, 'r')
            for key in loaded.keys():
                gaze_data.append(loaded[key])

        self.gaze_data = np.concatenate(gaze_data, axis=0)

    def train_from_torch(self, batch):
        obs = batch["observations"]
        actions = batch["actions"]
        labels = torch.argmax(actions, dim=-1)
        batch_size = obs.size(0) // 2
        gaze_obs = obs[:batch_size]
        prior_obs = obs[batch_size:]

        gaze_latent, gaze_pred = self.policy(gaze_obs, gaze=True)
        prior_latent, prior_pred = self.policy(prior_obs, gaze=False)

        pred = torch.cat((gaze_pred, prior_pred), dim=0)
        bc_loss = torch.nn.CrossEntropyLoss()(pred, labels)

        discrim_loss_fn = torch.nn.BCEWithLogitsLoss()
        gaze_loss = discrim_loss_fn(self.discrim(self.policy.shared_decoder(gaze_latent)), torch.zeros((gaze_latent.size(0), 1)).to(ptu.device))
        prior_loss = discrim_loss_fn(self.discrim(self.policy.shared_decoder(prior_latent)),
                                     0.9 * torch.ones((prior_latent.size(0), 1)).to(ptu.device))
        discrim_loss = (gaze_loss + prior_loss) / 2

        self.discrim_optimizer.zero_grad()
        discrim_loss.backward(retain_graph=True)
        self.discrim_optimizer.step()

        indices = np.random.choice(len(self.gaze_data), size=batch_size, replace=True)
        extra_gaze_batch = torch.from_numpy(self.gaze_data[indices]).to(ptu.device)
        extra_gaze_latent = self.policy.gaze_encoders[0](extra_gaze_batch)
        pred = self.policy.reconstructor(self.policy.shared_decoder(extra_gaze_latent))
        reconstruct_loss = self.policy.reconstruct_loss_fn(pred, extra_gaze_batch)

        combined_gaze_latent = torch.cat((extra_gaze_latent, gaze_latent), dim=0)

        aux_pred = self.discrim(self.policy.shared_decoder(combined_gaze_latent))
        aux_labels = torch.ones((combined_gaze_latent.size(0), 1)).to(ptu.device)
        aux_loss = torch.nn.BCEWithLogitsLoss()(aux_pred, aux_labels)

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
