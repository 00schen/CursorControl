import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import eval_np
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu


class VectorQuantizer(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self.encoder = Mlp(input_size=input_size,
                           output_size=embedding_dim,
                           hidden_sizes=hidden_sizes)

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, x, only_quantized=True):
        inputs = self.encoder(x)
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        embeddings = self._embedding.weight.to(flat_input.device)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(embeddings ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, embeddings.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, embeddings).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if only_quantized:
            return quantized
        else:
            return loss, quantized, perplexity, encodings


class VQGazePolicy(nn.Module):
    def __init__(self, input_size, output_size, encoder_hidden_sizes=(64,), decoder_hidden_sizes=(128, 128, 128, 128),
                 num_embeddings=10, embedding_dim=2, commitment_cost=0.25, layer_norm=False, gaze_dim=128):
        super(VQGazePolicy, self).__init__()

        self.vq_encoder = VectorQuantizer(gaze_dim, encoder_hidden_sizes, num_embeddings, embedding_dim,
                                          commitment_cost)

        self.decoder = Mlp(input_size=embedding_dim + input_size - gaze_dim,
                           output_size=output_size,
                           hidden_sizes=decoder_hidden_sizes,
                           layer_norm=layer_norm)
        self.gaze_dim = gaze_dim

    def forward(self, x):
        obs, gaze = x[..., :-self.gaze_dim], x[..., -self.gaze_dim:]
        loss, quantized, perplexity, _ = self.vq_encoder(gaze, only_quantized=False)
        obs = torch.cat((obs, quantized), dim=-1)
        pred = self.decoder(obs)

        return loss, pred, perplexity

    def get_action(self, x):
        return eval_np(self, x)[1], {}


class VAEGazePolicy(nn.Module):
    def __init__(self, input_size, output_size, encoder_hidden_sizes=(64,), decoder_hidden_sizes=(128, 128, 128, 128),
                 embedding_dim=1, layer_norm=False, gaze_dim=128, beta=1, output_activation=identity):
        super(VAEGazePolicy, self).__init__()

        self.encoder = Mlp(input_size=gaze_dim,
                           output_size=embedding_dim * 2,
                           hidden_sizes=encoder_hidden_sizes
                           )

        self.decoder = Mlp(input_size=embedding_dim + input_size - gaze_dim,
                           output_size=output_size,
                           hidden_sizes=decoder_hidden_sizes,
                           layer_norm=layer_norm,
                           output_activation=output_activation)
        self.gaze_dim = gaze_dim
        self.embedding_dim = embedding_dim
        self.beta = beta

    def forward(self, x, eps=0):
        obs, gaze = x[..., :-self.gaze_dim], x[..., -self.gaze_dim:]
        latent = self.encoder(gaze)
        mean, logvar = latent[..., :self.embedding_dim], latent[..., self.embedding_dim:]
        sample = mean + torch.exp(0.5 * logvar) * eps
        obs = torch.cat((obs, sample), dim=-1)
        pred = self.decoder(obs)
        kl_loss = -0.5 * (1 + logvar - torch.square(mean) - torch.exp(logvar))
        kl_loss = self.beta * torch.mean(kl_loss)

        return kl_loss, pred, sample

    def get_action(self, x):
        return eval_np(self, x)[1], {}

from numpy import prod
class TransferEncoderPolicy(nn.Module):
    # def __init__(self, ecnoder, decoder, decoder_pred_dim,
    #              gaze_dim=128, encoder_hidden_sizes=(64,64),
    #              layer_norm=False, beta=1, output_activation=identity):
    def __init__(self, encoder, decoder, beta=1):
        super().__init__()

        # self.encoder = Mlp(input_size=gaze_dim,
        #                    output_size=decoder_pred_dim,
        #                    hidden_sizes=encoder_hidden_sizes
        #                    )

        self.encoder = encoder
        self.decoder = decoder
        # self.layer_norm = ptu.zeros(1, requires_grad=True)
        # self.layer_norm = nn.LayerNorm(decoder.output_size)
        self.gaze_dim = encoder.input_size
        # self.decoder_pred_dim = decoder_pred_dim

    def forward(self, x, gaze=ptu.ones(1),):
        if len(x.shape) < 2:
            x = x[None,:]
        gaze_x = x[gaze.bool()]
        obs = x[torch.logical_not(gaze.bool())]
        if prod(gaze_x.size()):
            gaze_obs, gaze_feat = gaze_x[..., :-self.gaze_dim], gaze_x[..., -self.gaze_dim:]
            latent = self.encoder(gaze_feat)
            gaze_obs = torch.cat((gaze_obs, latent, ptu.zeros(latent.shape[0],self.gaze_dim-latent.shape[1])), dim=-1)
            if prod(obs.size()):
                obs = torch.cat((gaze_obs,obs))
            else:
                obs = gaze_obs
        pred = self.decoder(obs)
        # norm_pred = self.layer_norm(pred)
        x = x.squeeze()

        return pred

    def get_action(self, x):
        return eval_np(self, x), {}
