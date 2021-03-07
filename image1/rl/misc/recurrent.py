import torch as th
from torch import nn
from rlkit.torch.core import PyTorchModule
from rlkit.pythonplusplus import identity

class Recurrent(PyTorchModule):
	def __init__(self, input_size, hidden_size, output_size, num_layers, layer_norm=False, dropout=0, output_activation=identity):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.ln1 = nn.LayerNorm(input_size)
		self.lstm = nn.LSTM(input_size,hidden_size,num_layers=num_layers,batch_first=True,dropout=dropout)
		self.ln2 = nn.LayerNorm(hidden_size)
		self.last_fc = nn.Linear(hidden_size, output_size)
		self.output_activation = output_activation

	def forward(self, input, hidden_states=None):
		h = self.ln1(input)
		if hidden_states:
			h, new_hidden_states = self.lstm(h, hidden_states)
		else:
			h, new_hidden_states = self.lstm(h)
		h = self.ln2(h)
		h = self.last_fc(h)
		out = self.output_activation(h)
		return out, new_hidden_states
