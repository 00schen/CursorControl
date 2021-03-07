import torch as th
import os
from pathlib import Path
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule

class RewardEnsemble(PyTorchModule):
    def __init__(self,path):
        super().__init__()
        self.rfs = []
        p = Path(path)
        for child in p.iterdir():
            params_path = os.path.join(child,'params.pkl')
            rf = th.load(params_path,map_location=ptu.device)['trainer/rf']
            self.rfs.append(rf)

    def forward(self,*inputs):
        rewards = th.cat([rf(*inputs).detach() for rf in self.rfs],dim=1).sort(dim=1)[0][:,2]
        return rewards
