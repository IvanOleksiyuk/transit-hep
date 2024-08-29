from torch import nn
import wandb
from mltools.torch_utils import get_sched, get_loss_fn
from mltools.mlp import MLP
from kan import KAN
import torch 
from pytorch_lightning import LightningModule
from functools import partial
from typing import Any, Mapping
import torch.nn.functional as F
from torch.nn.functional import normalize, mse_loss, cosine_similarity
import numpy as np
import torch.distributed as dist
import twinturbo.src.models.distance_correlation as dcor
from twinturbo.src.models.pearson_correlation import PearsonCorrelation
import matplotlib.pyplot as plt
import PIL
import copy

def to_np(inpt) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a pytorch
    tensor to numpy array.

    - Includes gradient deletion, and device migration
    """
    if isinstance(inpt, (tuple, list)):
        return type(inpt)(to_np(x) for x in inpt)
    if inpt.dtype == torch.bfloat16:  # Numpy conversions don't support bfloat16s
        inpt = inpt.half()
    return inpt.detach().cpu().numpy()

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

        latent_dim: int,
        latent_norm: bool,
        optimizer: partial,
        scheduler: Mapping,
        encoder_mlp_config: Mapping,
        decoder_mlp_config: Mapping,
        var_group_list: list = None,
        loss_cfg: Mapping = None,
        use_m = True,
        input_noise_cfg=None,
        reverse_pass_mode=None,
        seed=42,
        valid_plots = True,
        adversarial_cfg = None,
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
        torch.manual_seed(seed)
        if adversarial_cfg is not None:
            if hasattr(adversarial_cfg, "mode"):
                self.adversarial = adversarial_cfg.mode
            else:
                self.adversarial = "default"
            self.automatic_optimization = False
            self.adversarial_cfg = adversarial_cfg
            self.discriminator = MLP(inpt_dim=latent_dim+1, outp_dim=1, **adversarial_cfg.discriminator)
            if not hasattr(adversarial_cfg, "g_loss_weight_in_warmup"):
                setattr(adversarial_cfg, "g_loss_weight_in_warmup", True)
            if not hasattr(adversarial_cfg, "train_dis_in_warmup"):
                setattr(adversarial_cfg, "train_dis_in_warmup", False)
            if not hasattr(adversarial_cfg, "loss_function"):
                setattr(adversarial_cfg, "loss_function", "binary_cross_entropy")
            if not hasattr(adversarial_cfg, "gradient_clip_val"):
                self.gradient_clip_val = 5
            else:
                self.gradient_clip_val = adversarial_cfg.gradient_clip_val
        else:
            self.adversarial = False
        self.use_m = use_m
        self.loss_cfg = loss_cfg
        self.valid_plots = valid_plots
        self.save_hyperparameters(logger=False)
        if encoder_mlp_config.get("network_type") == "KAN":
            self.encoder1 = KAN(width=[inpt_dim[0][0]]+encoder_mlp_config.hddn_dim+[latent_dim], grid=encoder_mlp_config.grid, k=encoder_mlp_config.k, device=torch.device('cuda')) #model = KAN(width=[2,5,1], grid=5, k=3, seed=0) 
            self.encoder2 = KAN(width=[inpt_dim[1][0]]+encoder_mlp_config.hddn_dim+[latent_dim], grid=encoder_mlp_config.grid, k=encoder_mlp_config.k, device=torch.device('cuda'))
            if self.use_m:
                self.decoder = KAN(width=[latent_dim*2]+decoder_mlp_config.hddn_dim+[inpt_dim[0][0]], grid=decoder_mlp_config.grid, k=decoder_mlp_config.k, device=torch.device('cuda'))
            else:
                self.decoder = KAN(width=[latent_dim*2]+decoder_mlp_config.hddn_dim+[inpt_dim[0][0]+inpt_dim[1][0]], grid=decoder_mlp_config.grid, k=decoder_mlp_config.k, device=torch.device('cuda'))

        else:
            self.encoder1 = MLP(inpt_dim=inpt_dim[0][0], outp_dim=latent_dim, **encoder_mlp_config)
            self.encoder2 = MLP(inpt_dim=inpt_dim[1][0], outp_dim=latent_dim, **encoder_mlp_config)
            if self.use_m:
                self.decoder = MLP(inpt_dim=latent_dim*2, outp_dim=inpt_dim[0][0], **decoder_mlp_config)
            else:
                self.decoder = MLP(inpt_dim=latent_dim*2, outp_dim=inpt_dim[0][0]+inpt_dim[1][0], **decoder_mlp_config)
        
        self.var_group_list = var_group_list
        self.latent_norm = latent_norm
        self.reverse_pass_mode = reverse_pass_mode
        self.input_noise_cfg = input_noise_cfg
        
        self.projector = self.get_projector(latent_dim, [32, 64, 128]) #TODO fix this 
        self.num_features = 2 # TODO fix this
        if hasattr(loss_cfg, "clip_loss_cfg") and loss_cfg.clip_loss_cfg is not None:
            self.clip_loss = CLIPLossNorm(loss_cfg.clip_loss_cfg.clip_logit_scale)
        if hasattr(loss_cfg, "DisCO_loss_cfg"):
            self.DisCO_loss = dcor.DistanceCorrelation()
        if hasattr(loss_cfg, "pearson_loss_cfg"):
            self.pearson_loss = PearsonCorrelation()
        if hasattr(loss_cfg, "vic_reg_cfg") and loss_cfg.vic_reg_cfg is not None:
            self.sim_coeff = loss_cfg.vic_reg_cfg.sim_coeff
            self.std_coeff = loss_cfg.vic_reg_cfg.std_coeff
            self.cov_coeff = loss_cfg.vic_reg_cfg.cov_coeff

        # For more stable checks in the shared step
        expected_attrs = ["reco", 
                        "consistency_x", 
                        "consistency_cont", 
                        "latent_variance_cfg", 
                        "l1_reg", 
                        "DisCO_loss_cfg", 
                        "pearson_loss_cfg", 
                        "clip_loss_cfg", 
                        "attractive", 
                        "repulsive", 
                        "vic_reg_cfg",
                        "loss_balancing",
                        "second_derivative_smoothness"]
        for attr in list(self.loss_cfg.keys()):
            if attr in expected_attrs:
                setattr(self.loss_cfg, attr, getattr(self.loss_cfg, attr, None))
            else:
                assert False, f"Unexpected loss config attribute: {attr}"
        for attr in expected_attrs:
            if not hasattr(self.loss_cfg, attr):
                setattr(self.loss_cfg, attr, None)

        if hasattr(self.loss_cfg.DisCO_loss_cfg, "w"):
            if not hasattr(self.loss_cfg.DisCO_loss_cfg, "mode"):
                self.loss_cfg.DisCO_loss_cfg.mode = "e1_vs_e2"

    def encode_e1_batch(self, batch):
        w1 = batch[0]
        if self.latent_norm:
            e1 = normalize(self.encoder1(w1))
        else:
            e1 = self.encoder1(w1)
        return e1

    def encode(self, w1, w2) -> torch.Tensor:
        if self.latent_norm:
            e1 = normalize(self.encoder1(w1))
            e2 = normalize(self.encoder2(w2))
        else:
            e1 = self.encoder1(w1)
            e2 = self.encoder2(w2)
        return e1, e2

    def encode_w1(self, w1) -> torch.Tensor:
        if self.latent_norm:
            e1 = normalize(self.encoder1(w1))
        else:
            e1 = self.encoder1(w1)
        return e1

    def encode_w2(self, w2) -> torch.Tensor:
        if self.latent_norm:
            e2 = normalize(self.encoder2(w2))
        else:
            e2 = self.encoder2(w2)
        return e2

    def VICloss(self, x, y):
        x = self.projector(x)
        y = self.projector(y)
        batch_size = x.size(0)
        repr_loss = F.mse_loss(x, y)

        #x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
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
        self.log(f"{step_type}_debug/global_step", self.global_step)
        batch_size=sample[0].shape[0]
        w1 = sample[0]
        w2 = sample[1]
        
        if self.input_noise_cfg is not None:
            w1[:, self.input_noise_cfg.w1mpos] += torch.randn_like(w1[:, self.input_noise_cfg.w1mpos]) * self.input_noise_cfg.noise_std_w1
            w2 += torch.randn_like(w2) * self.input_noise_cfg.noise_std_w2

        m_dn = w2
        if self.use_m:
            x = w1[:, :-w2.shape[1]]
        else:
            x = w1

        e1, e2 = self.encode(w1, w2)
        latent = torch.cat([e1, e2], dim=1)
        recon = self.decoder(latent)
        
        # Reverse pass
        rpm = torch.randperm(batch_size)
        if self.reverse_pass_mode is None:
            e2_p = e2[rpm]
        elif self.reverse_pass_mode == "noise":
            e2_p = e2[rpm] + self.reverse_pass_mode.noise_scale * torch.randn_like(e2)
        elif self.reverse_pass_mode == "additional_input":
            e2_p = self.encode_w2(sample[2])[rpm]
        else:
            e2_p = e2[rpm]

        latent_p = torch.cat([e1, e2_p], dim=1)
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
        total_loss = 0

        # Reconstruction loss
        loss_reco = mse_loss(recon, torch.cat([x, m_dn], dim=1)).mean()
        total_loss += loss_reco*self.loss_cfg.reco.w
        self.log(f"{step_type}/loss_reco", loss_reco)
        
        # Second derivative smoothmess
        if self.loss_cfg.second_derivative_smoothness is not None:
            e2_p_pl = self.encode_w2(sample[2] + self.loss_cfg.second_derivative_smoothness.step)[rpm]
            e2_p_mi = self.encode_w2(sample[2] - self.loss_cfg.second_derivative_smoothness.step)[rpm]
            latent_p_pl = torch.cat([e1, e2_p_pl], dim=1)
            recon_p_pl = self.decoder(latent_p_pl)
            #x_n_pl = recon_p_pl[:, :x.shape[1]]
            #m_n_pl = recon_p_pl[:, x.shape[1]:]
            latent_p_mi = torch.cat([e1, e2_p_mi], dim=1)
            recon_p_mi = self.decoder(latent_p_mi)
            #x_n_mi = recon_p_mi[:, :x.shape[1]]
            #m_n_mi = recon_p_mi[:, x.shape[1]:]	
            #loss_sec_der = (x_n_pl+x_n_mi-2*x_n)/(self.loss_cfg.second_derivative_smoothness.step**2) + (m_n_pl+m_n_mi-2*m_n)/(self.loss_cfg.second_derivative_smoothness.step**2)
            loss_sec_der = (recon_p_pl+recon_p_mi-2*recon_p)/(self.loss_cfg.second_derivative_smoothness.step**2)
            loss_sec_der = loss_sec_der.abs().mean()
            self.log(f"{step_type}/loss_sec_der_smooth", loss_sec_der)
            if self.loss_cfg.second_derivative_smoothness.w is not None:
                if isinstance(self.loss_cfg.second_derivative_smoothness.w, float) or isinstance(self.loss_cfg.second_derivative_smoothness.w, int):
                    total_loss += loss_sec_der*self.loss_cfg.second_derivative_smoothness.w
                else:
                    w = self.loss_cfg.second_derivative_smoothness.w(self.global_step)
                    total_loss += loss_sec_der*w
                    self.log(f"{step_type}_debug/second_derivative_smoothnessw", w)

        # Consistency losses 
        if self.loss_cfg.consistency_x is not None:
            loss_back_vec = mse_loss(e1, e1_n).mean()
            self.log(f"{step_type}/loss_back_vec", loss_back_vec)
            if self.loss_cfg.consistency_x.w is not None:
                if isinstance(self.loss_cfg.consistency_x.w, float) or isinstance(self.loss_cfg.consistency_x.w, int):
                    total_loss += loss_back_vec*self.loss_cfg.consistency_x.w
                else:
                    total_loss += loss_back_vec*self.loss_cfg.consistency_x.w(self.global_step)
        
        if self.loss_cfg.consistency_cont is not None:
            loss_back_cont = mse_loss(e2_p, e2_n).mean()
            self.log(f"{step_type}/loss_back_cont", loss_back_cont)
            if self.loss_cfg.consistency_cont.w is not None:
                if isinstance(self.loss_cfg.consistency_cont.w, float) or isinstance(self.loss_cfg.consistency_cont.w, int):
                    total_loss += loss_back_cont*self.loss_cfg.consistency_cont.w
                else:
                    total_loss += loss_back_cont*self.loss_cfg.consistency_cont.w(self.global_step)

        # variance loss
        if self.loss_cfg.latent_variance_cfg is not None:
            std_e1 = torch.sqrt(e1.var(dim=0) + 0.0001)
            std_e2 = torch.sqrt(e2.var(dim=0) + 0.0001)
            loss_latent_variance = torch.mean(torch.abs(1 - std_e1)**self.loss_cfg.latent_variance_cfg.pow) / 2 + torch.mean(torch.square(1 - std_e2)**self.loss_cfg.latent_variance_cfg.pow) / 2
            if self.loss_cfg.latent_variance_cfg.w is not None:
                total_loss += loss_latent_variance*self.loss_cfg.latent_variance_cfg.w
            self.log(f"{step_type}/variance_regularization", loss_latent_variance)

        # L1 regularization
        if self.loss_cfg.l1_reg is not None:
            all_params = torch.cat([x.view(-1) for x in self.parameters()])
            l1_regularization = self.loss_cfg.l1_reg*torch.norm(all_params, 1)
            total_loss += l1_regularization
            self.log(f"{step_type}/l1_regularization", l1_regularization)

        # Attractive and repulsive loss that are parts of triplet loss
        if self.loss_cfg.attractive is not None:
            loss_attractive = -cosine_similarity(e1, e2[torch.randperm(batch_size)]).mean()
            self.log(f"{step_type}/loss_attractive", loss_attractive)
            if self.loss_cfg.attractive.w is not None:
                if isinstance(self.loss_cfg.attractive.w, float) or isinstance(self.loss_cfg.attractive.w, int):
                    total_loss += loss_attractive*self.loss_cfg.attractive.w
                else:
                    total_loss += loss_attractive*self.loss_cfg.attractive.w(self.global_step)
        
        if self.loss_cfg.repulsive is not None:
            loss_repulsive = torch.abs(cosine_similarity(e1, e2)).mean()
            self.log(f"{step_type}/loss_repulsive", loss_repulsive)
            if self.loss_cfg.repulsive.w is not None:
                if isinstance(self.loss_cfg.repulsive.w, float) or isinstance(self.loss_cfg.repulsive.w, int):
                    total_loss += loss_repulsive*self.loss_cfg.repulsive.w
                else:
                    total_loss += loss_repulsive*self.loss_cfg.repulsive.w(self.global_step)
        # if hasattr(self.loss_cfg.loss_weights, "loss_attractive") and hasattr(self.loss_cfg.loss_weights, "loss_repulsive"):
        # 	self.log(f"{step_type}/loss_attractive+repulsive", loss_attractive+loss_repulsive)
        # 	# Loss balancing for tripplet loss
        # 	if step_type=="train" and hasattr(self.loss_cfg, "loss_balancing"):
        # 		self.loss_balabcing(loss_repulsive)
        # 		self.log("train/l_atr_weight", self.loss_cfg.loss_weights.loss_attractive)	
        # 		self.log("train/l_rep_weight", self.loss_cfg.loss_weights.loss_repulsive)

        # DisCO loss
        if self.loss_cfg.DisCO_loss_cfg is not None:
            if self.loss_cfg.DisCO_loss_cfg.mode == "e1_vs_e2":
                loss_disco = self.DisCO_loss(e1, e2)
            elif self.loss_cfg.DisCO_loss_cfg.mode == "e1_vs_w2":
                loss_disco = self.DisCO_loss(e1, w2)
            self.log(f"{step_type}/DisCO_loss", loss_disco)
            if self.loss_cfg.DisCO_loss_cfg.w is not None:
                if isinstance(self.loss_cfg.DisCO_loss_cfg.w, float) or isinstance(self.loss_cfg.DisCO_loss_cfg.w, int):
                        total_loss += loss_disco*self.loss_cfg.DisCO_loss_cfg.w
                else:
                    w = self.loss_cfg.DisCO_loss_cfg.w(self.global_step)
                    total_loss += loss_disco*w
                    self.log(f"{step_type}_debug/DisCO_lossw", w)

        # Pearson loss
        if self.loss_cfg.pearson_loss_cfg is not None:
            loss_disco = self.pearson_loss(e1, e2)
            self.log(f"{step_type}/pearson_loss", loss_disco)
            if self.loss_cfg.pearson_loss_cfg.w is not None:
                if isinstance(self.loss_cfg.pearson_loss_cfg.w, float) or isinstance(self.loss_cfg.pearson_loss_cfg.w, int):
                        total_loss += loss_disco*self.loss_cfg.pearson_loss_cfg.w
                else:
                    total_loss += loss_disco*self.loss_cfg.pearson_loss_cfg.w(self.global_step)

        # CLIP loss
        if self.loss_cfg.clip_loss_cfg is not None:
            loss_clip = self.clip_loss(e1, e2).mean()
            self.log(f"{step_type}/clip_loss", loss_clip)
            if self.loss_cfg.clip_loss_cfg.w is not None:
                total_loss += loss_clip*self.loss_cfg.clip_loss_cfg.w

        # VIC loss
        if self.loss_cfg.vic_reg_cfg is not None:
            loss, repr_loss, std_loss, cov_loss = self.VICloss(e1, e2)
            self.log(f"{step_type}/VIC_repr_loss", repr_loss)
            self.log(f"{step_type}/VIC_std_loss", std_loss)
            self.log(f"{step_type}/VIC_cov_loss", cov_loss)
            total_loss += loss

        # Log the total loss
        self.log(f"{step_type}/total_loss", total_loss)
        if self.adversarial:
            return total_loss, e1, w2
        else:
            return total_loss

    def loss_balabcing(self, loss_repulsive):
        if self.loss_cfg.loss_balancing==1:
            sum=self.loss_cfg.loss_weights.loss_repulsive+self.loss_cfg.loss_weights.loss_attractive
            if loss_repulsive>0.9:
                self.loss_cfg.loss_weights.loss_repulsive = self.loss_cfg.loss_weights.loss_repulsive*1.01
                self.loss_cfg.loss_weights.loss_attractive = self.loss_cfg.loss_weights.loss_attractive*0.99
                self.loss_cfg.loss_weights.loss_repulsive = sum*self.loss_cfg.loss_weights.loss_repulsive/(self.loss_cfg.loss_weights.loss_repulsive+self.loss_cfg.loss_weights.loss_attractive)
                self.loss_cfg.loss_weights.loss_attractive = sum*self.loss_cfg.loss_weights.loss_attractive/(self.loss_cfg.loss_weights.loss_repulsive+self.loss_cfg.loss_weights.loss_attractive)
            if loss_repulsive<0.1:
                self.loss_cfg.loss_weights.loss_repulsive = self.loss_cfg.loss_weights.loss_repulsive*0.99
                self.loss_cfg.loss_weights.loss_attractive = self.loss_cfg.loss_weights.loss_attractive*1.01
                self.loss_cfg.loss_weights.loss_repulsive = sum*self.loss_cfg.loss_weights.loss_repulsive/(self.loss_cfg.loss_weights.loss_repulsive+self.loss_cfg.loss_weights.loss_attractive)
                self.loss_cfg.loss_weights.loss_attractive = sum*self.loss_cfg.loss_weights.loss_attractive/(self.loss_cfg.loss_weights.loss_repulsive+self.loss_cfg.loss_weights.loss_attractive)
        if self.loss_cfg.loss_balancing==2:
            sum=self.loss_cfg.loss_weights.loss_repulsive+self.loss_cfg.loss_weights.loss_attractive
            if loss_repulsive>0.9:
                loss_repulsive_ = self.loss_cfg.loss_weights.loss_repulsive*1.1
                loss_attractive_ = self.loss_cfg.loss_weights.loss_attractive*0.9
                self.loss_cfg.loss_weights.loss_repulsive = sum*loss_repulsive_/(loss_repulsive_+loss_attractive_)
                self.loss_cfg.loss_weights.loss_attractive = sum*loss_attractive_/(loss_repulsive_+loss_attractive_)
            if loss_repulsive<0.1:
                loss_repulsive_ = self.loss_cfg.loss_weights.loss_repulsive*0.9
                loss_attractive_ = self.loss_cfg.loss_weights.loss_attractive*1.1
                self.loss_cfg.loss_weights.loss_repulsive = sum*loss_repulsive_/(loss_repulsive_+loss_attractive_)
                self.loss_cfg.loss_weights.loss_attractive = sum*loss_attractive_/(loss_repulsive_+loss_attractive_)
        if self.loss_balancing==3:
            sum=self.loss_cfg.loss_weights.loss_repulsive+self.loss_cfg.loss_weights.loss_attractive
            if loss_repulsive>0.9:
                self.loss_cfg.loss_weights.loss_repulsive = sum*3/4
                self.loss_cfg.loss_weights.loss_attractive = sum/4
            if loss_repulsive<0.1:
                self.loss_cfg.loss_weights.loss_repulsive = sum*1/4
                self.loss_cfg.loss_weights.loss_attractive = sum*3/4

    def adversarial_loss(self, y_hat, y):
        if self.adversarial_cfg.loss_function=="binary_cross_entropy":
            return F.binary_cross_entropy(y_hat, y.reshape((-1, 1))).mean()
        elif self.adversarial_cfg.loss_function=="mse":
            return mse_loss(y_hat, (y.reshape((-1, 1))-0.5)*2).mean()
        elif self.adversarial_cfg.loss_function=="hinge_loss":
            return F.hinge_embedding_loss(y_hat, (y.reshape((-1, 1))-0.5)*2).mean()

    def training_step(self, sample: tuple, batch_idx: int) -> torch.Tensor:
        if False: #self.adversarial=="latent+GAN":
            optimizer_g, optimizer_dl, optimizer_dg = self.optimizers()
            if self.lr_schedulers() is not None:
                for sched in self.lr_schedulers():
                    sched.step()
            # adversarial loss is binary cross-entropy

            total_loss, e1, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            batch_size=sample[0].shape[0]
            rpm = torch.randperm(batch_size)
            w2_perm = w2.clone()
            w2_perm = w2_perm[rpm]
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).type_as(w2_perm)
            e1_copy = e1.clone()
            generation = self.decoder(torch.cat([e1, self.encode_w2(w2_perm)], dim=1))
            trueth = sample[0]

            # train discriminator
            # Measure discriminator's ability to classify real from generated samples
            if self.current_epoch>self.adversarial_cfg.warmup:
                d_loss = self.adversarial_loss(self.discriminator_l(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                self.toggle_optimizer(optimizer_d)
                self.log("d_loss", d_loss, prog_bar=True)
                self.manual_backward(d_loss, retain_graph=True)
                optimizer_d.step()
                optimizer_d.zero_grad()
                self.untoggle_optimizer(optimizer_d)
    
                dg_loss = self.adversarial_loss(self.discriminator_g(torch.cat([generation, trueth], dim=0)), labels)
                self.toggle_optimizer(optimizer_dg)
                self.log("dg_loss", dg_loss, prog_bar=True)
                self.manual_backward(dg_loss, retain_graph=True)
                optimizer_dg.step()
                optimizer_dg.zero_grad()
                self.untoggle_optimizer(optimizer_dg)

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or self.global_step%self.adversarial_cfg.every_n_steps_g==0:
                g_loss = - self.adversarial_loss(self.discriminator_l(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                dg_loss = self.adversarial_lossself.discriminator_g(torch.cat([generation, trueth], dim=0), labels)
                total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight + dg_loss*self.adversarial_cfg.g_loss_weight
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=5)
                optimizer_g.step()
                optimizer_g.zero_grad()
                self.untoggle_optimizer(optimizer_g)

        if self.adversarial=="latent_assymetric":
            optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            batch_size=sample[0].shape[0]
            rpm = torch.randperm(batch_size)
            w2_perm = w2.clone()
            w2_perm = w2_perm[rpm]
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).type_as(w2_perm)
            e1_copy = e1.clone()
            # train discriminator
            # Measure discriminator's ability to classify real from generated samples
            if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                d_loss_1 = self.adversarial_loss(self.discriminator(torch.cat([e1, w2], dim=1)), torch.ones(batch_size).type_as(w2_perm))
                d_loss_0 = self.adversarial_loss(self.discriminator(torch.cat([e1_copy, w2_perm], dim=1)), torch.zeros(batch_size).type_as(w2_perm))
                d_loss = (d_loss_1 + d_loss_0)/2
                self.toggle_optimizer(optimizer_d)
                self.log("d_loss", d_loss, prog_bar=True)
                self.manual_backward(d_loss) #retain_graph=True
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                optimizer_d.zero_grad()
                self.untoggle_optimizer(optimizer_d)

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or self.global_step%self.adversarial_cfg.every_n_steps_g==0:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    g_loss_1 = - self.adversarial_loss(self.discriminator(torch.cat([e1, w2], dim=1)), torch.ones(batch_size).type_as(w2_perm))
                    #g_loss_0 = - self.adversarial_loss(self.discriminator(torch.cat([e1_copy, w2_perm], dim=1)), torch.zeros(batch_size))
                    total_loss2 = total_loss + g_loss_1*self.adversarial_cfg.g_loss_weight
                else:
                    total_loss2 = total_loss
                self.log("g_loss_1", g_loss_1, prog_bar=False)
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                optimizer_g.zero_grad()
                self.untoggle_optimizer(optimizer_g)
        elif self.adversarial=="latent_gaussian":
            optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            batch_size=sample[0].shape[0]
            rpm = torch.randperm(batch_size)
            w2_perm = w2.clone()
            w2_perm = w2_perm[rpm]
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).type_as(w2_perm)
            e1_copy = e1.clone()
            # train discriminator
            # Measure discriminator's ability to classify real from generated samples
            if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                d_loss = self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, torch.normal(0, 1, size=e1.shape).to(e1.device)], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                self.toggle_optimizer(optimizer_d)
                self.log("d_loss", d_loss, prog_bar=True)
                self.manual_backward(d_loss, retain_graph=True)
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                optimizer_d.zero_grad()
                self.untoggle_optimizer(optimizer_d)

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or self.global_step%self.adversarial_cfg.every_n_steps_g==0:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    g_loss = - self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, torch.normal(0, 1, size=e1.shape).to(e1.device)], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight
                else:
                    total_loss2 = total_loss
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                optimizer_g.zero_grad()
                self.untoggle_optimizer(optimizer_g)
        elif self.adversarial:
            optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            batch_size=sample[0].shape[0]
            rpm = torch.randperm(batch_size)
            w2_perm = w2.clone()
            w2_perm = w2_perm[rpm]
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).type_as(w2_perm)
            e1_copy = e1.clone()
            # train discriminator
            # Measure discriminator's ability to classify real from generated samples
            if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                d_loss = self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                self.toggle_optimizer(optimizer_d)
                self.log("d_loss", d_loss, prog_bar=True)
                self.manual_backward(d_loss, retain_graph=True)
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                optimizer_d.zero_grad()
                self.untoggle_optimizer(optimizer_d)

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or self.global_step%self.adversarial_cfg.every_n_steps_g==0:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    g_loss = - self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight
                else:
                    total_loss2 = total_loss
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                optimizer_g.zero_grad()
                self.untoggle_optimizer(optimizer_g)


        else:	
            total_loss = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            return total_loss

    def _draw_event_transport_trajectories(self, w1_, var, var_name, masses=np.linspace(-4, 4, 201), max_traj=20):
        w1 = copy.deepcopy(w1_)[:max_traj]
        recons = []
        if self.adversarial:
            zs = []
        for m in masses:
            w2 = torch.tensor(m).unsqueeze(0).expand(w1.shape[0], 1).float().to(w1.device)
            e1, e2 = self.encode(w1, w2)
            latent = torch.cat([e1, e2], dim=1)
            recon = self.decoder(latent)

            recons.append(recon)
            if self.adversarial:
                zs.append(self.discriminator(torch.cat([e1, w2], dim=1)))
        if self.adversarial:
            vmin = min([float(z[:max_traj].min().cpu().detach().numpy()) for z in zs])
            vmax = max([float(z[:max_traj].max().cpu().detach().numpy()) for z in zs])
        plt.figure()
        if max_traj is None:
            max_traj = w1.shape[0]
        for i in range(max_traj):
            x=masses
            y = np.array([float(recon[i, var].cpu().detach().numpy()) for recon in recons])
            if self.adversarial:
                z = np.array([float(z[i].cpu().detach().numpy()) for z in zs])
                plt.plot(x, y, "black", zorder=i*2+1)
                plt.scatter(x, y, c=z, cmap="turbo", s=2, zorder=i*2+2, vmin=vmin, vmax=vmax)
                if i==0:
                    plt.colorbar()
            else:
                plt.plot(x, y, "r")

        for i in range(max_traj):
            plt.scatter(to_np(w1[:, -1])[:max_traj], to_np(w1[:, var])[:max_traj],  marker="x", label="originals", c="green")
        plt.xlabel("mass")
        plt.ylabel(f"dim{var}")
        plt.title(f"Event transport for {var_name}, global step: {self.global_step}")
        # Convert to an image and return
        fig = plt.gcf()
        fig.tight_layout()
        fig.canvas.draw()
        img = PIL.Image.frombytes(
            "RGB",
            fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb(),
        )
        plt.close("all")
        return img

    def _draw_event_transport_trajectories_2nd_der(self, w1_, var, var_name, masses=None, max_traj=20):
        w1 = copy.deepcopy(w1_)[:max_traj]
        if masses is None:
            if self.loss_cfg.second_derivative_smoothness is not None:
                masses = np.arange(-4, 4, self.loss_cfg.second_derivative_smoothness.step)
                mode=" with step"
            else:
                masses = np.linspace(-4, 4, 81)
                mode="801"
        else:
            mode="custom"
        recons = []
        for m in masses:
            w2 = torch.tensor(m).unsqueeze(0).expand(w1.shape[0], 1).float().to(w1.device)
            e1, e2 = self.encode(w1, w2)
            latent = torch.cat([e1, e2], dim=1)
            recon = self.decoder(latent)

            recons.append(recon)
        
        plt.figure()
        if max_traj is None:
            max_traj = w1.shape[0]
        for i in range(max_traj):
            x = masses
            y = np.array([float(recon[i, var].cpu().detach().numpy()) for recon in recons])
            plt.plot(x[1:-1], (-2*y[1:-1]+y[:-2]+y[2:])/(x[1]-x[0])**2, "r")

        plt.xlabel("mass")
        plt.ylabel(f"dim{var}")
        plt.title(f"Event transport for {var_name}, global step: {self.global_step},"+mode)
        # Convert to an image and return
        fig = plt.gcf()
        fig.tight_layout()
        fig.canvas.draw()
        img = PIL.Image.frombytes(
            "RGB",
            fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb(),
        )
        plt.close("all")
        return img

    def validation_step (self, sample: tuple, batch_idx: int) -> torch.Tensor:
        total_loss = self._shared_step(sample, step_type="valid", _batch_index=batch_idx)
        if batch_idx == 0 and self.valid_plots:
            w1 = sample[0]
            for var in range(w1.shape[1]):
                image = wandb.Image(self._draw_event_transport_trajectories(w1, var=var, var_name=self.var_group_list[0][var], max_traj=20))
                if wandb.run is not None:
                    wandb.run.log({f"valid_images/transport_{self.var_group_list[0][var]}": image})
                image = wandb.Image(self._draw_event_transport_trajectories_2nd_der(w1, var=var, var_name=self.var_group_list[0][var], max_traj=20))
                if wandb.run is not None:
                    wandb.run.log({f"valid_images/transport_2nd_der_{self.var_group_list[0][var]}": image})


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
        if self.adversarial:	
            enc_dec_params = list(self.encoder1.parameters()) + list(self.encoder2.parameters()) + list(self.decoder.parameters())
            opt_g = self.hparams.optimizer(params=enc_dec_params)
            opt_d = self.hparams.optimizer(params=self.discriminator.parameters())
            if getattr(self.adversarial_cfg, "scheduler", None) is None:
                return [opt_g, opt_d], []
            elif self.adversarial_cfg.scheduler == "same_given":
                sched_g = self.hparams.scheduler.scheduler(opt_g)
                sched_d = self.hparams.scheduler.scheduler(opt_d)
                return [opt_g, opt_d], [sched_g, sched_d]
        else:
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
        if self.adversarial:
            if self.lr_schedulers() is not None:
                for sched in self.lr_schedulers():
                    sched.step()
    
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


