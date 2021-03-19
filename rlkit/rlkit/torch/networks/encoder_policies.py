import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import eval_np
import numpy as np
import h5py
import rlkit.torch.pytorch_util as ptu


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
                 embedding_dim=1, layer_norm=False, gaze_dim=128, beta=1):
        super(VAEGazePolicy, self).__init__()

        self.encoder = Mlp(input_size=gaze_dim,
                           output_size=embedding_dim * 2,
                           hidden_sizes=encoder_hidden_sizes
                           )

        self.decoder = Mlp(input_size=embedding_dim + input_size - gaze_dim,
                           output_size=output_size,
                           hidden_sizes=decoder_hidden_sizes,
                           layer_norm=layer_norm)
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

        return kl_loss, pred

    def get_action(self, x):
        return eval_np(self, x)[1], {}


class MixedPolicy(nn.Module):
    def __init__(self, input_size, output_size, encoder_hidden_sizes=(32,), decoder_hidden_sizes=(128, 128, 128, 128),
                 embedding_dim=1, layer_norm=False, gaze_dim=128, num_encoders=3):
        super(MixedPolicy, self).__init__()

        self.gaze_encoders = []
        for i in range(num_encoders):
            encoder = Mlp(input_size=gaze_dim,
                          output_size=embedding_dim,
                          hidden_sizes=encoder_hidden_sizes)
            self.__setattr__("gaze_encoder{}".format(i), encoder)
            self.gaze_encoders.append(encoder)

        # self.one_hot_encoder = Mlp(input_size=gaze_dim,
        #                            output_size=embedding_dim * 2,
        #                            hidden_sizes=[]
        #                            )

        self.decoder = Mlp(input_size=embedding_dim + input_size - gaze_dim,
                           output_size=output_size,
                           hidden_sizes=decoder_hidden_sizes,
                           layer_norm=layer_norm)

        self.auto_decoder = Mlp(input_size=embedding_dim, output_size=gaze_dim, hidden_sizes=encoder_hidden_sizes)

        self.gaze_dim = gaze_dim
        self.embedding_dim = embedding_dim
        self.num_encoders = num_encoders

        data_paths = ['image/rl/gaze_capture/gaze_data_train.h5']
        gaze_data = []
        for path in data_paths:
            loaded = h5py.File(path, 'r')
            for key in loaded.keys():
                gaze_data.append(loaded[key])

        self.gaze_data = np.concatenate(gaze_data, axis=0)
        #self.gaze_data = np.concatenate([data[key][()] for key in data.keys()], axis=0)
        self.reconstruct_loss_fn = torch.nn.MSELoss()

    def forward(self, x, gaze=True, train=True):
        obs, aux = x[..., :-self.gaze_dim], x[..., -self.gaze_dim:]
        reconstruct_loss = 0
        if gaze:
            if train:
                encoder = np.random.choice(self.gaze_encoders)
                latent = encoder(aux)
                indices = np.random.choice(len(self.gaze_data), size=obs.size(0), replace=True)
                gaze_batch = torch.from_numpy(self.gaze_data[indices]).to(ptu.device)
                gaze_latent = encoder(gaze_batch)
                pred = self.auto_decoder(gaze_latent)
                reconstruct_loss = self.reconstruct_loss_fn(pred, gaze_batch)
                latent_ret = torch.cat((latent, gaze_latent), dim=0)
            else:
                latent = 0
                for encoder in self.gaze_encoders:
                    latent += encoder(aux)
                latent = latent / self.num_encoders

                latent_ret = latent
        else:
            latent = aux[..., :self.embedding_dim]
            latent_ret = latent

        # mean, logvar = latent[..., :self.embedding_dim], latent[..., self.embedding_dim:]
        # sample = mean #+ torch.exp(0.5 * logvar) * eps

        obs = torch.cat((obs, latent), dim=-1)
        pred = self.decoder(obs)
        # kl_loss = -0.5 * (1 + logvar - torch.square(mean) - torch.exp(logvar))
        # kl_loss = self.beta * torch.mean(kl_loss)

        return latent_ret, pred, reconstruct_loss

    def get_action(self, x):
        return eval_np(self, x, train=False)[1], {}
