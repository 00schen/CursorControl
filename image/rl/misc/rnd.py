from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp
import torch.optim as optim

class RNDModel:
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = hparams['obs_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        self.f = Mlp(
			input_size= self.obs_dim,
			output_size=self.output_size,
			hidden_sizes=[self.size]*self.n_layers,
            hidden_init=lambda weight: weight.data.uniform_()
		)
        self.f_hat = Mlp(
			input_size= self.obs_dim,
			output_size=self.output_size,
			hidden_sizes=[self.size]*self.n_layers,
            hidden_init=lambda weight: weight.data.normal_()
		)
        
        self.optimizer = optim.Adam(
			self.f.parameters(),
			lr=1e-3,
			weight_decay=1e-5,
		)

        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def get_target(self,torch):
        error = torch.norm(self.f(ob_no).detach()-self.f_hat(ob_no),dim=1)
        return error

    def update(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        loss = self(ob_no).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()