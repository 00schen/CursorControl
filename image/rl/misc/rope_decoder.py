import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch
from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp, NeuralProcessMlp
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
from rlkit.torch.core import elem_or_tuple_to_numpy
from rlkit.torch.distributions import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class RopeNeuralProcessMlp(NeuralProcessMlp):
    def forward(self, input):
        if len(input.size()) < 2:
            set_dim=0
        else:
            set_dim=1
        input = torch.cat(input.unsqueeze(set_dim).split(self.input_size, -1), dim=set_dim)
        return super().forward(input,set_dim=set_dim)

class RopeDecoder(PyTorchModule):
    def __init__(
        self,
        rope_latent_size,
        rope_input_indices,
        rope_input_size,
        other_obs_input_size,
        output_size,
        hidden_sizes,
        **kwargs
    ):
        super().__init__()
        self.rope_input_indices = rope_input_indices
        self.np = RopeNeuralProcessMlp(
            input_size=rope_input_size,
            output_size=rope_latent_size,
            hidden_sizes=[hidden_sizes[0]],
            **kwargs)
        self.mlp = Mlp(
            input_size=rope_latent_size + other_obs_input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            **kwargs
        )
    
    def forward(self, *inputs):
        """ Assumes observation is first input"""
        mlp_input, np_input = inputs[0].tensor_split(self.rope_input_indices, dim=-1)
        np_latents = self.np(np_input)
        mlp_input = torch.cat((np_latents, mlp_input, *inputs[1:]),dim=-1)
        output = self.mlp(mlp_input)
        return output      
        
class RopeTanhGaussianPolicy(RopeDecoder, TorchStochasticPolicy):
    def forward(self, *inputs):
        mean, log_std = RopeDecoder.forward(self, *inputs).split(self.input_sizes//2, -1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob

class RopeConcatTanhGaussianPolicy(RopeTanhGaussianPolicy):
    def get_action(self, *obs_np):
        actions = self.get_actions(*[obs[None] for obs in obs_np])
        return actions[0, :], {}

    def get_actions(self, *obs_np):
        dist = self._get_dist_from_np(*obs_np)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)