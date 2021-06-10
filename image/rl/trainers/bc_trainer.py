import torch
import torch.optim as optim
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from .gan_trainer import TorchCycleGANSubTrainer
from collections import OrderedDict
import rlkit.torch.pytorch_util as ptu


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


class TorchEncDecAWACTrainer(TorchBCTrainer):
    def __init__(
            self,
            policy,
            policy_lr,
            qf1,
            target_qf1,
            qf2,
            target_qf2,
            qf_lr,
            vae,
            latent_size,
            optimizer_class=optim.Adam,
            beta=0.01,
            temp=0.3,
            discount=0.99,
            sample=True,
            soft_target_tau=1e-2,
            target_update_period=1,
    ):
        super().__init__(policy, policy_lr, optimizer_class)
        self.vae = vae
        self.qf1 = qf1
        self.target_qf1 = target_qf1
        self.qf2 = qf2
        self.target_qf2 = target_qf2
        self.qf1_optimizer = optimizer_class(self.qf1.parameters(), lr=qf_lr)
        self.qf2_optimizer = optimizer_class(self.qf2.parameters(), lr=qf_lr)
        self.encoder_optimizer = optimizer_class(
            self.vae.encoder.parameters(),
            lr=policy_lr,
        )
        self.beta = beta
        self.discount = discount
        self.latent_size = latent_size
        self.sample = sample
        self.temp = temp
        self.qf_criterion = torch.nn.MSELoss()
        self._n_train_steps_total = 0
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

    def train_from_torch(self, batch):
        obs = batch['observations']
        rewards = batch['rewards']
        terminals = batch['terminals']
        actions = batch['actions']
        next_obs = batch['next_observations']
        curr_goal = batch['curr_goal']
        next_goal = batch['next_goal']
        curr_goal_set = batch.get('curr_goal_set')
        next_goal_set = batch.get('next_goal_set')

        batch_size = obs.shape[0]
        has_goal_set = curr_goal_set is not None

        eps = torch.normal(ptu.zeros((batch_size, self.latent_size)), 1) if self.sample else None
        curr_latent, kl_loss = self.vae.sample(curr_goal, eps=eps, return_kl=True)
        next_latent = self.vae.sample(next_goal, eps=eps, return_kl=False)

        next_latent = next_latent.detach()

        """
        Policy and Alpha Loss
        """

        if has_goal_set:
            curr_goal_set_flat = curr_goal_set.reshape((batch_size, -1))
            curr_policy_features = [obs, curr_goal_set_flat, curr_latent]
        else:
            curr_policy_features = [obs, curr_latent]

        dist = self.policy(*curr_policy_features)

        log_pi = dist.log_prob(actions)

        new_obs_actions, _ = dist.rsample_and_logprob()

        if has_goal_set:
            curr_qf_features = [obs, curr_goal_set_flat, curr_goal, actions]
            new_qf_features = [obs, curr_goal_set_flat, curr_goal, new_obs_actions]
            next_goal_set_flat = next_goal_set.reshape((batch_size, -1))
            next_policy_features = [next_obs, next_goal_set_flat, next_latent]
        else:
            curr_qf_features = [obs, curr_goal, actions]
            new_qf_features = [obs, curr_goal, new_obs_actions]
            next_policy_features = [next_obs, next_latent]

        q_estimates = self.qf1(*curr_qf_features)
        value_estimates = self.qf1(*new_qf_features)
        advantages = q_estimates - value_estimates
        policy_loss = -(log_pi * torch.exp(advantages / self.temp).detach().reshape(batch_size)).mean()

        q1_pred = self.qf1(*curr_qf_features)
        q2_pred = self.qf2(*curr_qf_features)
        next_dist = self.policy(*next_policy_features)

        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()

        if has_goal_set:
            next_qf_features = [next_obs, next_goal_set_flat, next_goal, new_next_actions]
        else:
            next_qf_features = [next_obs, next_goal, new_next_actions]

        target_q_values = torch.min(
            self.target_qf1(*next_qf_features),
            self.target_qf2(*next_qf_features),
        )

        q_target = rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        (policy_loss + self.beta * kl_loss).backward()
        self.policy_optimizer.step()
        self.encoder_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1
        self.try_update_target_networks()

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2, self.vae]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            vae=self.vae
        )


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
        super().__init__(policy, optimizer)
        self.gan_trainer = TorchCycleGANSubTrainer(encoder, **gan_kwargs)

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
        return [self.policy] + self.gan_trainer.networks

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
