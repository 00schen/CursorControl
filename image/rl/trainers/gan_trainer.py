import torch as th
from torch.nn import functional as F
import torch.optim as optim
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu


class TorchCycleGANSubTrainer(TorchTrainer):
	def __init__(
			self,
			encoder,
			lr=1e-3,
			optimizer_class=optim.Adam,
			gaze_recon_w = 1,
			target_pos_recon_w = 0,
			adversarial_w=1,
	):
		super().__init__()
		self.encoder = encoder
		feature_size = encoder.input_size
		self.encoder_inv = Mlp(hidden_sizes=[128,128],
							input_size=feature_size,
							output_size=feature_size,
							layer_norm=True)
		self.enc_disc = Mlp(hidden_sizes=[128,128],
							input_size=feature_size,
							output_size=1,
							layer_norm=True)
		self.enc_inv_disc = Mlp(hidden_sizes=[128,128],
							input_size=feature_size,
							output_size=1,
							layer_norm=True)
		self.optimizer = optimizer_class(
			list(self.encoder.parameters())+list(self.encoder_inv.parameters()),
			lr=lr,
		)
		self.disc_optimizer = optimizer_class(
			list(self.enc_disc.parameters())+list(self.enc_inv_disc.parameters()),
			lr=lr,
		)
		self.gaze_recon_w = gaze_recon_w
		self.target_pos_recon_w = target_pos_recon_w
		self.adversarial_w = adversarial_w

	def train_from_torch(self, batch):
		obs = batch["observations"][:,-128:]
		gaze = batch['gaze'].flatten()
		unstruct = batch['unstructured_gaze']
		x = obs[gaze.bool()] # assumes that gaze is about 50-50
		y = obs[th.logical_not(gaze.bool())]
		pred_y = self.encoder(x)
		unstr_pred_y = self.encoder(unstruct)
		pred_x = self.encoder_inv(y)
		recon_x = self.encoder_inv(th.cat((pred_y,unstr_pred_y)))
		recon_y = self.encoder(pred_x)

		# measure l1 loss of reconstruction
		x_recon_loss = F.l1_loss(recon_x,th.cat((x,unstruct)))
		y_recon_loss = F.l1_loss(recon_y,y)
		
		# measure adversarial loss of reconstruction
		y_target = ptu.ones(y.shape[0])
		pred_y_target = ptu.zeros(pred_y.shape[0])
		target_y = th.cat((y_target,pred_y_target))

		x_target = ptu.ones(x.shape[0])
		pred_x_target = ptu.zeros(pred_x.shape[0])
		target_x = th.cat((x_target,pred_x_target))

		disc_pred_y_min = self.enc_disc(th.cat((y,pred_y)).detach())
		disc_pred_x_min = self.enc_inv_disc(th.cat((x,pred_x)).detach())
		adv_loss_y_min = F.binary_cross_entropy_with_logits(disc_pred_y_min.flatten(),target_y)
		adv_loss_x_min = F.binary_cross_entropy_with_logits(disc_pred_x_min.flatten(),target_x)

		disc_pred_y_max = self.enc_disc(th.cat((y,pred_y)))
		disc_pred_x_max = self.enc_inv_disc(th.cat((x,pred_x)))
		adv_loss_y_max = F.binary_cross_entropy_with_logits(disc_pred_y_max.flatten(),target_y)
		adv_loss_x_max = F.binary_cross_entropy_with_logits(disc_pred_x_max.flatten(),target_x)

		loss = self.gaze_recon_w*x_recon_loss + self.target_pos_recon_w*y_recon_loss - self.adversarial_w*(adv_loss_x_max+adv_loss_y_max)
		disc_loss = adv_loss_x_min+adv_loss_y_min
		# def optimizer_step():
		self.optimizer.zero_grad()
		self.enc_disc.requires_grad = False
		self.enc_inv_disc.requires_grad = False
		loss.backward()
		self.optimizer.step()
		self.disc_optimizer.zero_grad()
		self.enc_disc.requires_grad = True
		self.enc_inv_disc.requires_grad = True
		disc_loss.backward(retain_graph=True)
		self.disc_optimizer.step()

		eval_stats = dict(
			gaze_recon_loss=x_recon_loss.item(),
			target_pos_recon_loss=y_recon_loss.item(),
			gaze_adv_loss=adv_loss_x_min.item(),
			target_pos_adv_loss=adv_loss_y_min.item(),
		)		

		return eval_stats

	@property
	def networks(self):
		return [self.encoder,self.encoder_inv,self.enc_disc,self.enc_inv_disc]

	def get_snapshot(self):
		return dict(
			encoder=self.encoder,
			encoder_inv=self.encoder_inv,
			enc_disc=self.enc_disc,
			enc_inv_disc=self.enc_inv_disc,
		)