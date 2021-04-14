import torch
import h5py
import random
import math
from datetime import datetime
import os
from rlkit.torch.networks import VAE
from rlkit.pythonplusplus import identity


hidden_sizes = (64,)
embedding_dim = 3
gaze_dim = 128
epochs = 100
lr = 1e-3
batch_size = 128
beta = 0.1
time = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')
save_path = 'image/logs/gaze_vae/'
os.makedirs(save_path, exist_ok=True)

data_paths = ['image/rl/gaze_capture/gaze_data.h5']
gaze_data = []
for path in data_paths:
    loaded = h5py.File(path, 'r')
    for key in loaded.keys():
        gaze_data.extend(loaded[key])

gaze_vae = VAE(hidden_sizes, hidden_sizes, latent_size=embedding_dim, input_size=gaze_dim)
optimizer = torch.optim.Adam(gaze_vae.parameters(), lr=lr)

n_batches = math.ceil(len(gaze_data) / batch_size)
for i in range(epochs):
    random.shuffle(gaze_data)
    for j in range(n_batches):
        batch = torch.Tensor(gaze_data[j * batch_size: (j + 1) * batch_size])
        x_reconstruct, (reconstruct_loss, kl_loss) = gaze_vae(batch)
        loss = reconstruct_loss + beta * kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(reconstruct_loss, kl_loss)
print(torch.mean(torch.std(batch, dim=0)))
print(torch.mean(torch.std(x_reconstruct, dim=0)))
torch.save(gaze_vae, os.path.join(save_path, time + '.pkl'))






