import torch as th
import torch.nn as nn
import torch.nn.functional as F
from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import eval_np

class TrainedVAEGazePolicy(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, eps=0):
        latent = self.encoder(x)
        mean, logvar = th.split(latent,self.encoder.output_size//2,dim=1)
        eps = th.normal(mean=th.zeros(mean.shape))
        sample = mean + th.exp(0.5 * logvar) * eps
        output = self.decoder(sample)

        return output

    def get_action(self, x):
        return eval_np(self, x), {}