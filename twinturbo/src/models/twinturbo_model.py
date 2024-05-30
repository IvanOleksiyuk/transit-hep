from torch import nn
import wandb
from mltools.torch_utils import get_sched, get_loss_fn
from mltools.mlp import MLP
import torch 
from pytorch_lightning import LightningModule
from functools import partial
from typing import Any, Mapping
import torch.nn.functional as F
from torch.nn.functional import normalize, mse_loss, cosine_similarity
import numpy as np
import torch.distributed as dist

class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=1.0):
        super().__init__()
        self.logit_scale = logit_scale

    def forward(self, embedding_1, embedding_2, valid=False):
        device = embedding_1.device
        logits_1 = self.logit_scale * embedding_1 @ embedding_2.T
        logits_2 = self.logit_scale * embedding_2 @ embedding_1.T
        num_logits = logits_1.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        loss = 0.5 * (
            F.cross_entropy(logits_1, labels) + F.cross_entropy(logits_2, labels)
        )
        return loss

class CLIPLossNorm(nn.Module):
    def __init__(
        self,
        logit_scale_init=np.log(1 / 0.07),
        logit_scale_max=np.log(100),
        logit_scale_min=np.log(0.01),
        logit_scale_learnable=False,
    ):
        super().__init__()
        if logit_scale_learnable:
            self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        else:
            self.logit_scale = torch.ones([], requires_grad=False) * logit_scale_init
        self.logit_scale_max = logit_scale_max
        self.logit_scale_min = logit_scale_min
        self.logit_scale_learnable = logit_scale_learnable

    def forward(self, embedding_1, embedding_2, valid=False):
        scale = torch.clamp(
            self.logit_scale, max=self.logit_scale_max, min=self.logit_scale_min
        ).exp()

        wandb.log({"scale": scale})
        device = embedding_1.device
        norm = (
            embedding_1.norm(dim=1, keepdim=True)
            @ embedding_2.norm(dim=1, keepdim=True).T
        )
        logits_1 = (scale * embedding_1 @ embedding_2.T) / norm
        logits_2 = (scale * embedding_2 @ embedding_1.T) / norm.T
        num_logits = logits_1.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        loss = 0.5 * (
            F.cross_entropy(logits_1, labels) + F.cross_entropy(logits_2, labels)
        )
        return loss

class FullGatherLayer(torch.autograd.Function):
	"""
	Gather tensors from all process and support backward propagation
	for the gradients across processes.
	"""

	@staticmethod
	def forward(ctx, x):
		output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
		dist.all_gather(output, x)
		return tuple(output)

	@staticmethod
	def backward(ctx, *grads):
		all_gradients = torch.stack(grads)
		dist.all_reduce(all_gradients)
		return all_gradients[dist.get_rank()]

def off_diagonal(x):
	n, m = x.shape
	assert n == m
	return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class TwinTURBO(LightningModule):
    
	def __init__(
		self,
		*,
		inpt_dim: int,
		encoder_mlp_config: Mapping,
		decoder_mlp_config: Mapping,
		latent_dim: int,
		latent_norm: bool,
		optimizer: partial,
		scheduler: Mapping,
		var_group_list: list = None,
  		loss_weights: Mapping = None,

  		l1_reg = 0.0,
    	use_m = True,
    	loss_balancing= None,

    	clip_loss_cfg: Mapping = None,
		vic_reg_cfg=None,
		input_noise_cfg=None,
	) -> None:
		"""
		Args:
			inpt_dim: Number of edge, node and high level features
			normaliser_config: Config for the IterativeNormLayer
			optimizer: Partially initialised optimiser
			scheduler: How the sceduler should be used
		"""
		super().__init__()
		# TODO need to preprocess the data properly!
		self.use_m = use_m
		self.loss_weights=loss_weights
		self.save_hyperparameters(logger=False)
		self.encoder1 = MLP(inpt_dim=inpt_dim[0][0], outp_dim=latent_dim, **encoder_mlp_config)
		self.encoder2 = MLP(inpt_dim=inpt_dim[1][0], outp_dim=latent_dim, **encoder_mlp_config)
		if self.use_m:
			self.decoder = MLP(inpt_dim=latent_dim*2, outp_dim=inpt_dim[0][0], **decoder_mlp_config)
		else:
			self.decoder = MLP(inpt_dim=latent_dim*2, outp_dim=inpt_dim[0][0]+inpt_dim[1][0], **decoder_mlp_config)

		self.var_group_list = var_group_list
		self.latent_norm = latent_norm

		self.input_noise_cfg = input_noise_cfg
		self.l1_reg = l1_reg
		self.loss_balancing = loss_balancing
		self.clip_loss_cfg = clip_loss_cfg
		if clip_loss_cfg is not None:
			self.use_clip = True
			self.clip_loss = CLIPLossNorm(clip_loss_cfg.clip_logit_scale)
		else:
			self.use_clip = False

		if vic_reg_cfg is not None:
			self.sim_coeff = vic_reg_cfg.sim_coeff
			self.std_coeff = vic_reg_cfg.std_coeff
			self.cov_coeff = vic_reg_cfg.cov_coeff
		
		self.projector = self.get_projector(latent_dim, [32, 64, 128]) #TODO fix this 
		self.batch_size = 512 #TODO fix this 
		self.num_features = 2 # TODO fix this
		self.vic_reg_cfg= vic_reg_cfg

	def encode(self, w1, w2) -> torch.Tensor:
		if self.latent_norm:
			e1 = normalize(self.encoder1(w1))
			e2 = normalize(self.encoder2(w2))
		else:
			e1 = self.encoder1(w1)
			e2 = self.encoder2(w2)
		return e1, e2

	def VICloss(self, x, y):
		x = self.projector(x)
		y = self.projector(y)

		repr_loss = F.mse_loss(x, y)

		#x = torch.cat(FullGatherLayer.apply(x), dim=0)
		#y = torch.cat(FullGatherLayer.apply(y), dim=0)
		x = x - x.mean(dim=0)
		y = y - y.mean(dim=0)

		std_x = torch.sqrt(x.var(dim=0) + 0.0001)
		std_y = torch.sqrt(y.var(dim=0) + 0.0001)
		std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

		cov_x = (x.T @ x) / (self.batch_size - 1)
		cov_y = (y.T @ y) / (self.batch_size - 1)
		cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
			self.num_features
		) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

		loss = (
			self.sim_coeff * repr_loss
			+ self.std_coeff * std_loss
			+ self.cov_coeff * cov_loss
		)
		return loss, repr_loss, std_loss, cov_loss

	def get_projector(self, embedding, mlp):
		layers = []
		f = [embedding] + mlp
		for i in range(len(f) - 2):
			layers.append(nn.Linear(f[i], f[i + 1]))
			layers.append(nn.BatchNorm1d(f[i + 1]))
			layers.append(nn.ReLU(True))
		layers.append(nn.Linear(f[-2], f[-1], bias=False))
		return nn.Sequential(*layers)

	def _shared_step(self, sample: tuple, _batch_index = None, step_type="none") -> torch.Tensor:
		batch_size=sample[0].shape[0]
		w1, w2 = sample
		
		if self.input_noise_cfg is not None:
			w1 = w1[:, self.input_noise_cfg.w1mpos] + torch.randn_like(w1[:, self.input_noise_cfg.w1mpos]) * self.noise_cfg.noise_std_w1
			w2 = w2 + torch.randn_like(w2) * self.noise_cfg.noise_std_w2

		m_dn = w2
		if self.use_m:
			x = w1[:, :-w2.shape[1]]
		else:
			x = w1

		e1, e2 = self.encode(w1, w2)
		latent = torch.cat([e1, e2], dim=1)
		recon = self.decoder(latent)
		
		# Reverse pass
		e1_p = e1[torch.randperm(batch_size)]
		latent_p = torch.cat([e1_p, e2], dim=1)
		recon_p = self.decoder(latent_p)
		x_n = recon_p[:, :x.shape[1]]
		m_n = recon_p[:, x.shape[1]:]
		
		if self.use_m:
			w1_n = torch.cat([x_n, m_n*self.use_m], dim=1)
		else:
			w1_n = x_n
		w2_n = m_n

		e1_n, e2_n = self.encode(w1_n, w2_n)

		#### Losses
		# Reconstruction loss
		loss_reco = mse_loss(recon, torch.cat([x, m_dn], dim=1)).mean()
		# Consistency loss (also backwards loss)
		loss_back_vec = mse_loss(e1_p, e1_n).mean()
		loss_back_cont = mse_loss(e2, e2_n).mean()
		# Attractive and repulsive loss that are parts of triplet loss
		loss_attractive = -cosine_similarity(e1, e2[torch.randperm(batch_size)]).mean()
		loss_repulsive = torch.abs(cosine_similarity(e1, e2)).mean()
		# L1 regularization
		all_params = torch.cat([x.view(-1) for x in self.parameters()])
		l1_regularization = self.l1_reg*torch.norm(all_params, 1)
  
		total_loss = sum([l1_regularization,
					loss_reco*self.loss_weights.loss_reco,
					loss_attractive*self.loss_weights.loss_attractive,
					loss_repulsive*self.loss_weights.loss_repulsive,
					loss_back_vec*self.loss_weights.loss_back_vec,
					loss_back_cont*self.loss_weights.loss_back_cont])
		self.log(f"{step_type}/total_loss", total_loss)
		self.log(f"{step_type}/loss_reco", loss_reco)
		self.log(f"{step_type}/loss_attractive", loss_attractive)
		self.log(f"{step_type}/loss_attractive+repulsive", loss_attractive+loss_repulsive)
		self.log(f"{step_type}/loss_repulsive", loss_repulsive)
		self.log(f"{step_type}/loss_back_vec", loss_back_vec)
		self.log(f"{step_type}/loss_back_cont", loss_back_cont)
		self.log(f"{step_type}/l1_regularization", l1_regularization)
		if self.use_clip:
			loss_clip = self.clip_loss(e1, e2).mean()
			self.log(f"{step_type}/clip_loss", loss_clip)
			total_loss += loss_clip*self.clip_loss_cfg.clip_loss_weight
   
		if self.vic_reg_cfg is not None:
			loss, repr_loss, std_loss, cov_loss = self.VICloss(e1, e2)
			self.log(f"{step_type}/VIC_repr_loss", repr_loss)
			self.log(f"{step_type}/VIC_std_loss", std_loss)
			self.log(f"{step_type}/VIC_cov_loss", cov_loss)
			total_loss += loss
	
		if step_type=="train":
			self.loss_balabcing(loss_repulsive)
		self.log("train/l_atr_weight", self.loss_weights.loss_attractive)	
		self.log("train/l_rep_weight", self.loss_weights.loss_repulsive)
		return total_loss

	def loss_balabcing(self, loss_repulsive):
		if self.loss_balancing==1:
			sum=self.loss_weights.loss_repulsive+self.loss_weights.loss_attractive
			if loss_repulsive>0.9:
				self.loss_weights.loss_repulsive = self.loss_weights.loss_repulsive*1.01
				self.loss_weights.loss_attractive = self.loss_weights.loss_attractive*0.99
				self.loss_weights.loss_repulsive = sum*self.loss_weights.loss_repulsive/(self.loss_weights.loss_repulsive+self.loss_weights.loss_attractive)
				self.loss_weights.loss_attractive = sum*self.loss_weights.loss_attractive/(self.loss_weights.loss_repulsive+self.loss_weights.loss_attractive)
			if loss_repulsive<0.1:
				self.loss_weights.loss_repulsive = self.loss_weights.loss_repulsive*0.99
				self.loss_weights.loss_attractive = self.loss_weights.loss_attractive*1.01
				self.loss_weights.loss_repulsive = sum*self.loss_weights.loss_repulsive/(self.loss_weights.loss_repulsive+self.loss_weights.loss_attractive)
				self.loss_weights.loss_attractive = sum*self.loss_weights.loss_attractive/(self.loss_weights.loss_repulsive+self.loss_weights.loss_attractive)
		if self.loss_balancing==2:
			sum=self.loss_weights.loss_repulsive+self.loss_weights.loss_attractive
			if loss_repulsive>0.9:
				loss_repulsive_ = self.loss_weights.loss_repulsive*1.1
				loss_attractive_ = self.loss_weights.loss_attractive*0.9
				self.loss_weights.loss_repulsive = sum*loss_repulsive_/(loss_repulsive_+loss_attractive_)
				self.loss_weights.loss_attractive = sum*loss_attractive_/(loss_repulsive_+loss_attractive_)
			if loss_repulsive<0.1:
				loss_repulsive_ = self.loss_weights.loss_repulsive*0.9
				loss_attractive_ = self.loss_weights.loss_attractive*1.1
				self.loss_weights.loss_repulsive = sum*loss_repulsive_/(loss_repulsive_+loss_attractive_)
				self.loss_weights.loss_attractive = sum*loss_attractive_/(loss_repulsive_+loss_attractive_)
		if self.loss_balancing==3:
			sum=self.loss_weights.loss_repulsive+self.loss_weights.loss_attractive
			if loss_repulsive>0.9:
				self.loss_weights.loss_repulsive = sum*3/4
				self.loss_weights.loss_attractive = sum/4
			if loss_repulsive<0.1:
				self.loss_weights.loss_repulsive = sum*1/4
				self.loss_weights.loss_attractive = sum*3/4

	def training_step(self, sample: tuple, batch_idx: int) -> torch.Tensor:
		total_loss = self._shared_step(sample, step_type="train")
		return total_loss

	def validation_step (self, sample: tuple, batch_idx: int) -> torch.Tensor:
		total_loss = self._shared_step(sample, step_type="valid")
		return total_loss

	def on_fit_start(self, *_args) -> None:
		"""Function to run at the start of training."""
		# Define the metrics for wandb (otherwise the min wont be stored!)
		if wandb.run is not None:
			for step_type in ["train", "valid"]:
				wandb.define_metric(f"{step_type}/total_loss", summary="min")

	def configure_optimizers(self) -> dict:
		"""Configure the optimisers and learning rate sheduler for this
		model."""

		# Finish initialising the partialy created methods
		opt = self.hparams.optimizer(params=self.parameters())

		sched = self.hparams.scheduler.scheduler(opt)

		# Return the dict for the lightning trainer
		return {
			"optimizer": opt,
			"lr_scheduler": {"scheduler": sched, **self.hparams.scheduler.lightning},
		}

	def on_validation_epoch_end(self) -> None:
		"""Makes several plots of the jets and how they are reconstructed.
		"""

	def generate(self, sample: tuple) -> torch.Tensor:
		w1, w2 = sample
		if self.latent_norm:
			e1 = normalize(self.encoder1(w1))
			e2 = normalize(self.encoder2(w2))
		else:
			e1 = self.encoder1(w1)
			e2 = self.encoder2(w2)
		latent = torch.cat([e1, e2], dim=1)
		recon = self.decoder(latent)
		return recon

	def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
		if len(batch)==2:
			context = batch[1]
			sample = self.generate(batch).squeeze(1)
			#context = context.repeat(1, n_over).reshape(-1, 1)
			sample = sample.reshape(-1, sample.shape[-1])
			# Return dict of data, sample and log prob
			result = {var_name: column.reshape(-1, 1) for var_name, column in zip(self.var_group_list[0], sample.T)}
			result.update({var_name: column.reshape(-1, 1) for var_name, column in zip(self.var_group_list[1], context.T)})
			return result

