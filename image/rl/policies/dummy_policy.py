import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.env_utils import get_dim

class DummyPolicy(PyTorchModule):
	def __init__(self, env):
		super().__init__()
		self.action_size = get_dim(env.action_space)

	def get_action(self, obs):
		action = np.zeros(self.action_size)
		action[0] = 1
		return action, {}

	def reset(self):
		pass