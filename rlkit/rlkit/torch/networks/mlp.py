import torch
from torch import nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.networks import LayerNorm
from rlkit.torch.pytorch_util import activation_from_string


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class MultiHeadedMlp(Mlp):
    """
                   .-> linear head 0
                  /
    input --> MLP ---> linear head 1
                  \
                   .-> linear head 2
    """

    def __init__(
            self,
            hidden_sizes,
            output_sizes,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activations=None,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=sum(output_sizes),
            input_size=input_size,
            init_w=init_w,
            hidden_activation=hidden_activation,
            hidden_init=hidden_init,
            b_init_value=b_init_value,
            layer_norm=layer_norm,
            layer_norm_kwargs=layer_norm_kwargs,
        )
        self._splitter = SplitIntoManyHeads(
            output_sizes,
            output_activations,
        )

    def forward(self, input):
        flat_outputs = super().forward(input)
        return self._splitter(flat_outputs)


class ConcatMultiHeadedMlp(MultiHeadedMlp):
    """
    Concatenate inputs along dimension and then pass through MultiHeadedMlp.
    """

    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class ConcatMlp(Mlp):
    """
    Concatenate inputs along dimension and then pass through MLP.
    """

    def __init__(self, *args, dim=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpQf(ConcatMlp):
    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            action_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer

    def forward(self, obs, actions, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        if self.action_normalizer:
            actions = self.action_normalizer.normalize(actions)
        return super().forward(obs, actions, **kwargs)


class MlpQfWithObsProcessor(Mlp):
    def __init__(self, obs_processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_processor = obs_processor

    def forward(self, obs, actions, **kwargs):
        h = self.obs_processor(obs)
        flat_inputs = torch.cat((h, actions), dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpGoalQfWithObsProcessor(Mlp):
    def __init__(self, obs_processor, obs_dim,
                 backprop_into_obs_preprocessor=True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_processor = obs_processor
        self.backprop_into_obs_preprocessor = backprop_into_obs_preprocessor
        self.obs_dim = obs_dim

    def forward(self, obs, actions, **kwargs):
        h_s = self.obs_processor(obs[:, :self.obs_dim])
        h_g = self.obs_processor(obs[:, self.obs_dim:])
        if not self.backprop_into_obs_preprocessor:
            h_s = h_s.detach()
            h_g = h_g.detach()
        flat_inputs = torch.cat((h_s, h_g, actions), dim=1)
        return super().forward(flat_inputs, **kwargs)


class SplitIntoManyHeads(nn.Module):
    """
           .-> head 0
          /
    input ---> head 1
          \
           '-> head 2
    """

    def __init__(
            self,
            output_sizes,
            output_activations=None,
    ):
        super().__init__()
        if output_activations is None:
            output_activations = ['identity' for _ in output_sizes]
        else:
            if len(output_activations) != len(output_sizes):
                raise ValueError("output_activation and output_sizes must have "
                                 "the same length")

        self._output_narrow_params = []
        self._output_activations = []
        for output_activation in output_activations:
            if isinstance(output_activation, str):
                output_activation = activation_from_string(output_activation)
            self._output_activations.append(output_activation)
        start_idx = 0
        for output_size in output_sizes:
            self._output_narrow_params.append((start_idx, output_size))
            start_idx = start_idx + output_size

    def forward(self, flat_outputs):
        pre_activation_outputs = tuple(
            flat_outputs.narrow(1, start, length)
            for start, length in self._output_narrow_params
        )
        outputs = tuple(
            activation(x)
            for activation, x in zip(
                self._output_activations, pre_activation_outputs
            )
        )
        return outputs


class ParallelMlp(nn.Module):
    """
    Efficient implementation of multiple MLPs with identical architectures.

           .-> mlp 0
          /
    input ---> mlp 1
          \
           '-> mlp 2

    See https://discuss.pytorch.org/t/parallel-execution-of-modules-in-nn-modulelist/43940/7
    for details

    The last dimension of the output corresponds to the MLP index.
    """

    def __init__(
            self,
            num_heads,
            input_size,
            output_size_per_mlp,
            hidden_sizes,
            hidden_activation='relu',
            output_activation='identity',
            input_is_already_expanded=False,
    ):
        super().__init__()

        def create_layers():
            layers = []
            input_dim = input_size
            for i, hidden_size in enumerate(hidden_sizes):
                fc = nn.Conv1d(
                    in_channels=input_dim * num_heads,
                    out_channels=hidden_size * num_heads,
                    kernel_size=1,
                    groups=num_heads,
                )
                layers.append(fc)
                if isinstance(hidden_activation, str):
                    activation = activation_from_string(hidden_activation)
                else:
                    activation = hidden_activation
                layers.append(activation)
                input_dim = hidden_size

            last_fc = nn.Conv1d(
                in_channels=input_dim * num_heads,
                out_channels=output_size_per_mlp * num_heads,
                kernel_size=1,
                groups=num_heads,
            )
            layers.append(last_fc)
            if output_activation != 'identity':
                if isinstance(output_activation, str):
                    activation = activation_from_string(output_activation)
                else:
                    activation = output_activation
                layers.append(activation)
            return layers

        self.network = nn.Sequential(*create_layers())
        self.num_heads = num_heads
        self.input_is_already_expanded = input_is_already_expanded

    def forward(self, x):
        if not self.input_is_already_expanded:
            x = x.repeat(1, self.num_heads).unsqueeze(-1)
        flat = self.network(x)
        batch_size = x.shape[0]
        return flat.view(batch_size, -1, self.num_heads)


class MlpGazePolicy(PyTorchModule):
    def __init__(self,
                 encoder_hidden_sizes,
                 decoder_hidden_sizes,
                 output_size,
                 input_size,
                 init_w=3e-3,
                 hidden_activation=F.relu,
                 output_activation=identity,
                 hidden_init=ptu.fanin_init,
                 b_init_value=0,
                 layer_norm=False,
                 layer_norm_kwargs=None,
                 gaze_dim=128,
                 embedding_dim=3,
                 gaze_vae=None,
                 decoder=None,
                 rew_classifier=None
                 ):
        super().__init__()
        self.latent_dim = embedding_dim
        if decoder is None:
            self.decoder = Mlp(hidden_sizes=decoder_hidden_sizes, output_size=output_size,
                               input_size=input_size - gaze_dim + self.latent_dim,
                               init_w=init_w, hidden_activation=hidden_activation, output_activation=output_activation,
                               hidden_init=hidden_init, b_init_value=b_init_value, layer_norm=layer_norm,
                               layer_norm_kwargs=layer_norm_kwargs)
        else:
            self.decoder = decoder
        if gaze_vae is None:
            self.gaze_vae = VAE(encoder_hidden_sizes, encoder_hidden_sizes,
                                latent_size=self.latent_dim, input_size=gaze_dim)
        else:
            self.gaze_vae = gaze_vae
        self.gaze_dim = gaze_dim
        if rew_classifier is None:
            self.rew_classifier = Mlp(hidden_sizes=decoder_hidden_sizes, output_size=1,
                                      input_size=input_size - gaze_dim + self.latent_dim,
                                      init_w=init_w, hidden_activation=hidden_activation,
                                      output_activation=output_activation,
                                      hidden_init=hidden_init, b_init_value=b_init_value, layer_norm=layer_norm,
                                      layer_norm_kwargs=layer_norm_kwargs)
        else:
            self.rew_classifier = rew_classifier

    def forward(self, obs, eps=None, return_kl=False, skip_encoder=False, **kwargs):
        if not skip_encoder:
            # assumes gaze at end of obs
            h, gaze = obs[..., :-self.gaze_dim], obs[..., -self.gaze_dim:]
            sample, kl_loss = self.gaze_vae.sample(gaze, eps, return_kl=True)
            h = torch.cat((h, sample), dim=-1)
        else:
            h = obs
            kl_loss = None

        if return_kl:
            return self.decoder(h, **kwargs), kl_loss
        return self.decoder(h, **kwargs)

    def rew_classification(self, obs, eps=None, return_kl=True, train_encoder=True):
        h, gaze = obs[..., :-self.gaze_dim], obs[..., -self.gaze_dim:]
        sample, kl_loss = self.gaze_vae.sample(gaze, eps, return_kl=True)
        h = torch.cat((h, sample), dim=-1)

        if not train_encoder:
            h = h.detach()
            kl_loss = 0

        if return_kl:
            return self.rew_classifier(h), kl_loss
        return self.rew_classifier(h)

    def get_actions(self, obs, skip_encoder=False):
        return eval_np(self, obs, return_kl=False, skip_encoder=skip_encoder)

    def get_action(self, obs_np, skip_encoder=False):
        actions = self.get_actions(obs_np[None], skip_encoder=skip_encoder)
        return actions[0, :], {}


class MlpVQVAEGazePolicy(MlpGazePolicy):
    def __init__(self,
                 encoder_hidden_sizes,
                 decoder_hidden_sizes,
                 output_size,
                 input_size,
                 init_w=3e-3,
                 hidden_activation=F.relu,
                 output_activation=identity,
                 hidden_init=ptu.fanin_init,
                 b_init_value=0,
                 layer_norm=False,
                 layer_norm_kwargs=None,
                 gaze_dim=128,
                 embedding_dim=1,
                 gaze_vae=None,
                 decoder=None,
                 rew_classifier=None,
                 n_embed_per_latent=10,
                 n_latents=3
                 ):
        super().__init__(encoder_hidden_sizes=encoder_hidden_sizes,
                         decoder_hidden_sizes=decoder_hidden_sizes,
                         output_size=output_size,
                         input_size=input_size,
                         init_w=init_w,
                         hidden_activation=hidden_activation,
                         output_activation=output_activation,
                         hidden_init=hidden_init,
                         b_init_value=b_init_value,
                         layer_norm=layer_norm,
                         layer_norm_kwargs=layer_norm_kwargs,
                         gaze_dim=gaze_dim,
                         embedding_dim=embedding_dim * n_latents,
                         gaze_vae=gaze_vae,
                         decoder=decoder,
                         rew_classifier=rew_classifier
                         )
        if gaze_vae is None:
            self.gaze_vae = VQVAE(encoder_hidden_sizes, encoder_hidden_sizes,
                                  latent_size=embedding_dim, input_size=gaze_dim, n_embed_per_latent=n_embed_per_latent,
                                  n_latents=n_latents)
        else:
            self.gaze_vae = gaze_vae
        if rew_classifier is None:
            self.rew_classifier = Mlp(hidden_sizes=decoder_hidden_sizes, output_size=1,
                                      input_size=input_size - gaze_dim + self.latent_dim,
                                      init_w=init_w, hidden_activation=hidden_activation,
                                      output_activation=output_activation,
                                      hidden_init=hidden_init, b_init_value=b_init_value, layer_norm=layer_norm,
                                      layer_norm_kwargs=layer_norm_kwargs)
        else:
            self.rew_classifier = rew_classifier

    def forward(self, obs, return_vq=False, skip_encoder=False, **kwargs):
        if not skip_encoder:
            # assumes gaze at end of obs
            h, gaze = obs[..., :-self.gaze_dim], obs[..., -self.gaze_dim:]
            sample, vq_loss1, vq_loss2 = self.gaze_vae.sample(gaze, return_vq=True)
            h = torch.cat((h, sample), dim=-1)
        else:
            h = obs
            vq_loss1, vq_loss2 = 0, 0

        if return_vq:
            return self.decoder(h, **kwargs), vq_loss1, vq_loss2
        return self.decoder(h, **kwargs)

    def rew_classification(self, obs, eps=None, return_vq=True, train_encoder=True):
        h, gaze = obs[..., :-self.gaze_dim], obs[..., -self.gaze_dim:]
        sample, vq_loss1, vq_loss2 = self.gaze_vae.sample(gaze, return_vq=True)
        h = torch.cat((h, sample), dim=-1)

        if not train_encoder:
            h = h.detach()
            vq_loss1, vq_loss2 = 0, 0

        if return_vq:
            return self.rew_classifier(h), vq_loss1, vq_loss2
        return self.rew_classifier(h)

    def get_actions(self, obs, skip_encoder=False):
        return eval_np(self, obs, return_vq=False, skip_encoder=skip_encoder)


class VAE(PyTorchModule):
    def __init__(
            self,
            encoder_hidden_sizes,
            decoder_hidden_sizes,
            latent_size,
            input_size,
            output_activation=F.relu,
    ):
        super().__init__()
        self.encoder = Mlp(hidden_sizes=encoder_hidden_sizes, output_size=latent_size * 2, input_size=input_size)
        self.decoder = Mlp(hidden_sizes=decoder_hidden_sizes, output_size=input_size, input_size=latent_size,
                           output_activation=output_activation)
        self.latent_size = latent_size

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_reconstruct = self.decode(z)
        return x_reconstruct, self.compute_loss(x, x_reconstruct, mean, logvar)

    def compute_loss(self, x, x_reconstruct, mean, logvar):
        reconstruct_loss = torch.nn.MSELoss()(x, x_reconstruct)
        return reconstruct_loss, self.kl_loss(mean, logvar)

    def kl_loss(self, mean, logvar):
        return torch.mean(-0.5 * (1 + logvar - torch.square(mean) - torch.exp(logvar)))

    def encode(self, x):
        mean, logvar = torch.split(self.encoder(x), self.latent_size, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = torch.normal(torch.zeros(self.latent_size).to(ptu.device), 1)
        return mean + eps * torch.exp(logvar * 0.5)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, x, eps=None, return_kl=True):
        mean, logvar = torch.split(self.encoder(x), self.latent_size, dim=1)
        sample = mean
        if eps is not None:
            sample = sample + torch.exp(logvar * 0.5) * eps
        if return_kl:
            return sample, self.kl_loss(mean, logvar)
        return sample


class VQVAE(PyTorchModule):
    def __init__(
            self,
            encoder_hidden_sizes,
            decoder_hidden_sizes,
            latent_size,
            input_size,
            output_activation=F.relu,
            n_latents=3,
            n_embed_per_latent=10,
    ):
        super().__init__()
        self.encoder = Mlp(hidden_sizes=encoder_hidden_sizes, output_size=latent_size * n_latents, input_size=input_size)
        self.decoder = Mlp(hidden_sizes=decoder_hidden_sizes, output_size=input_size, input_size=latent_size * n_latents,
                           output_activation=output_activation)
        self.n_latents = n_latents
        self.latent_size = latent_size
        self.n_embed_per_latent = n_embed_per_latent
        self.embedding = nn.Embedding(self.n_embed_per_latent, self.latent_size)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed_per_latent, 1.0 / self.n_embed_per_latent)

    def forward(self, x):
        z = self.encode(x)
        z_q = self.quantize(z)
        x_reconstruct = self.decode(z_q)
        return x_reconstruct, self.compute_loss(x, x_reconstruct, z, z_q)

    def compute_loss(self, x, x_reconstruct, z, z_q):
        reconstruct_loss = torch.nn.MSELoss()(x, x_reconstruct)
        return (reconstruct_loss,) + self.vq_loss(z, z_q)

    def vq_loss(self, z, z_q):
        return torch.mean((z_q.detach() - z) ** 2), torch.mean((z_q - z.detach()) ** 2)

    def encode(self, x):
        return torch.reshape(self.encoder(x), (-1, self.n_latents, self.latent_size))

    def decode(self, z_q):
        return self.decoder(torch.reshape(z_q, (-1, self.n_latents * self.latent_size)))

    def quantize(self, z):
        z_flat = torch.reshape(z, (-1, self.latent_size))

        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flat, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_embed_per_latent).to(ptu.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        return z_q

    def sample(self, x, return_vq=True):
        z = self.encode(x)
        z_q = self.quantize(z)
        z_q_flat = torch.reshape(z_q, (-1, self.n_latents * self.latent_size))
        if return_vq:
            return (z_q_flat,) + self.vq_loss(z, z_q)
        return z_q_flat


class QrMlp(Mlp):
    def __init__(
            self,
            hidden_sizes,
            action_size,
            input_size,
            atom_size=200,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__(hidden_sizes=hidden_sizes, output_size=action_size * atom_size, input_size=input_size,
                         init_w=init_w, hidden_activation=hidden_activation, output_activation=output_activation,
                         hidden_init=hidden_init, b_init_value=b_init_value, layer_norm=layer_norm,
                         layer_norm_kwargs=layer_norm_kwargs)
        self.action_size = action_size
        self.atom_size = atom_size

    def forward(self, input, return_preactivations=False):
        output = super().forward(input, return_preactivations)
        preactivation = None
        if return_preactivations:
            output, preactivation = output

        logits = torch.reshape(output, [-1, self.action_size, self.atom_size])
        q_values = torch.mean(logits, dim=2)

        if preactivation is None:
            return logits, q_values

        else:
            return logits, q_values, preactivation

    def get_action(self, input):
        return eval_np(self, input, return_preactivations=False)[1], {}


class QrGazeMlp(QrMlp):
    def __init__(
            self,
            encoder_hidden_sizes,
            decoder_hidden_sizes,
            action_size,
            input_size,
            atom_size=200,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
            gaze_dim=128,
            embedding_dim=3,
    ):
        super().__init__(hidden_sizes=decoder_hidden_sizes, action_size=action_size,
                         input_size=input_size - gaze_dim + embedding_dim, atom_size=atom_size,
                         init_w=init_w, hidden_activation=hidden_activation, output_activation=output_activation,
                         hidden_init=hidden_init, b_init_value=b_init_value, layer_norm=layer_norm,
                         layer_norm_kwargs=layer_norm_kwargs)
        self.gaze_encoder = Mlp(hidden_sizes=encoder_hidden_sizes, output_size=embedding_dim, input_size=gaze_dim)
        self.gaze_dim = gaze_dim
        self.embedding_dim = embedding_dim

    def forward(self, input, return_preactivations=False):
        gaze, h = input[..., -self.gaze_dim:], input[..., :-self.gaze_dim]
        latent = self.gaze_encoder(gaze)
        h = torch.cat((latent, h), dim=-1)

        return super().forward(h, return_preactivations)

    def get_action(self, input):
        return eval_np(self, input, return_preactivations=False)[1], {}
