import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")


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
from mattstools.mattstools.simple_transformers import TransformerEncoder, FullEncoder, TransformerVectorEncoder
from mltools.modules import IterativeNormLayer

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

class DiscPC(nn.Module):
    """A generative model which uses the diffusion process on a point cloud."""

    def __init__(
        self,
        inpt_dim,
        ctxt_dim,
        #normaliser_config: Mapping,
        architecture: partial,
        final_mlp: partial,
    ) -> None:
        super().__init__()

        # Attributes
        self.pc_dim = inpt_dim 
        self.hlv_dim = ctxt_dim

        # Normalisation layers
        #self.norm = IterativeNormLayer(self.pc_dim, **normaliser_config)
        #self.hlv_norm = IterativeNormLayer(self.hlv_dim, **normaliser_config)
        # self.mjj_norm = IterativeNormLayer(self.mjj_dim, **normaliser_config)

        # The shared layer transformer
        self.trans = architecture(
            inpt_dim=self.pc_dim,
            ctxt_dim=self.hlv_dim,
        )

        # The final mlp for determining source of the data
        self.final_mlp = final_mlp(
            inpt_dim=self.trans.outp_dim,
            outp_dim=1,
            # ctxt_dim=self.mjj_dim,
        )

    def forward(self, pc, ctxt, mask=None) -> torch.Tensor:
        """Pass through the network."""


        # Normalise everything
        #jet1 = self.norm(jet1, mask)
        #hlv1 = self.hlv_norm(hlv1)
        # mjj = self.mjj_norm(mjj)

        # Process each jet seperately and sum (order invariant)
        x = self.trans(pc, mask, ctxt)
        
        # Pass through the final mlp
        return self.final_mlp(x)



class TwinTURBO(LightningModule):
    
    def __init__(
        self,
        *,
        inpt_dim: int,

        latent_dim: int,
        latent_norm: bool,
        optimizer: partial,
        scheduler: Mapping,
        encoder_cfg,
        decoder_cfg,
        network_type = "MLP",
        encoder_cfg2: Mapping = None,
        latent_dim2: int = None, 
        var_group_list: list = None,
        loss_cfg: Mapping = None,
        use_m_encodig = True,
        input_noise_cfg=None,
        reverse_pass_mode=None,
        seed=42,
        valid_plots = True,
        adversarial_cfg = None,
        discriminator_latent_cfg = None,
        add_standardizing_layer = False,
        afterglow_epoch = np.inf,
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
        self.afterglow_epoch = afterglow_epoch
        if adversarial_cfg is not None:
            if hasattr(adversarial_cfg, "mode"):
                self.adversarial = adversarial_cfg.mode
            else:
                self.adversarial = "default"
            self.automatic_optimization = False
            self.adversarial_cfg = adversarial_cfg

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
        self.use_m_encodig = use_m_encodig
        self.loss_cfg = loss_cfg
        self.valid_plots = valid_plots
        self.save_hyperparameters(logger=False)
        self.network_type = network_type
        print("SELECTED NETWORK TYPE: ", network_type)
        self.latent_norm_enc1 = latent_norm
        self.latent_norm_enc2 = latent_norm
        
        # Initialise the networks
        x_dim = inpt_dim[0][0]
        context_dim = inpt_dim[1][0]
        self.second_input_mask = False #By default
        
        if network_type == "Transformer_conditional":
            self.second_input_mask = True
            self.encoder1 = FullEncoder(inpt_dim=x_dim, outp_dim=latent_dim, ctxt_dim=context_dim, **encoder_cfg)
            self.encoder2 = lambda x: x
            self.decoder = FullEncoder(inpt_dim=latent_dim, outp_dim=x_dim, ctxt_dim=context_dim, **decoder_cfg)
            if self.adversarial:
                self.discriminator = DiscPC(inpt_dim=latent_dim, ctxt_dim=context_dim, **self.adversarial_cfg.discriminator)
                if "double_discriminator" in self.adversarial:
                    self.discriminator2 = DiscPC(inpt_dim=x_dim, ctxt_dim=context_dim, **self.adversarial_cfg.discriminator2)
                
            # if discriminator_latent_cfg is not None:
            #     self.discriminator = TransformerEncoder(inpt_dim=latent_dim, outp_dim=1, **discriminator_latent_cfg)
            self.style_injection_cond = True
            self.decoder_out_m = False
        elif network_type == "partial_context":
            self.decoder_out_m = False
            self.encoder1 = encoder_cfg(inpt_dim=x_dim, ctxt_dim=inpt_dim[1][0], outp_dim=latent_dim)
            self.decoder = decoder_cfg(inpt_dim=latent_dim, ctxt_dim=inpt_dim[1][0], outp_dim=x_dim)
            self.style_injection_cond = True
            if self.adversarial:
                self.discriminator = adversarial_cfg.discriminator(inpt_dim=latent_dim, outp_dim=1, ctxt_dim=context_dim)
                if "double_discriminator" in self.adversarial:
                    self.discriminator2 = adversarial_cfg.discriminator2(inpt_dim=x_dim, outp_dim=1, ctxt_dim=context_dim)
            self.encoder2 = lambda x: x
        elif network_type == "MLP_cont":
            self.decoder_out_m = False
            self.encoder1 = MLP(inpt_dim=x_dim, ctxt_dim=inpt_dim[1][0], outp_dim=latent_dim, **encoder_cfg)
            self.decoder = MLP(inpt_dim=latent_dim, ctxt_dim=inpt_dim[1][0], outp_dim=x_dim, **decoder_cfg)
            self.style_injection_cond = True
            if self.adversarial:
                self.discriminator = MLP(inpt_dim=latent_dim, outp_dim=1, ctxt_dim=context_dim, **adversarial_cfg.discriminator)
                if "double_discriminator" in self.adversarial:
                    self.discriminator2 = MLP(inpt_dim=x_dim, outp_dim=1, ctxt_dim=context_dim, **adversarial_cfg.discriminator2)
        elif network_type == "KAN":
            self.encoder1 = KAN(width=[x_dim]+encoder_cfg.hddn_dim+[latent_dim], grid=encoder_cfg.grid, k=encoder_cfg.k, device=torch.device('cuda')) #model = KAN(width=[2,5,1], grid=5, k=3, seed=0) 
            self.encoder2 = KAN(width=[inpt_dim[1][0]]+encoder_cfg.hddn_dim+[latent_dim], grid=encoder_cfg.grid, k=encoder_cfg.k, device=torch.device('cuda'))
            decoder_out_dim = x_dim+inpt_dim[1][0]
            self.decoder_out_m = True
            self.style_injection_cond = False
            if self.use_m_encodig:
                self.decoder = KAN(width=[latent_dim*2]+decoder_cfg.hddn_dim+[x_dim], grid=decoder_cfg.grid, k=decoder_cfg.k, device=torch.device('cuda'))
            else:
                self.decoder = KAN(width=[latent_dim*2]+decoder_cfg.hddn_dim+[decoder_out_dim], grid=decoder_cfg.grid, k=decoder_cfg.k, device=torch.device('cuda'))
            if self.adversarial:
                self.discriminator = MLP(inpt_dim=latent_dim+1, outp_dim=1, **adversarial_cfg.discriminator)
                if "double_discriminator" in self.adversarial:
                    self.discriminator2 = MLP(inpt_dim=x_dim+inpt_dim[1][0], outp_dim=1, **adversarial_cfg.discriminator2)
        elif network_type == "MLP_no_m_encoding":
            self.encoder1 = MLP(inpt_dim=x_dim+inpt_dim[1][0], outp_dim=latent_dim, **encoder_cfg)
            latent_dim2 = 1
            self.encoder2 = lambda x: x
            self.decoder_out_m = False
            self.decoder = MLP(inpt_dim=latent_dim+latent_dim2, outp_dim=x_dim, **decoder_cfg) 
            self.latent_norm_enc2 = False
            self.style_injection_cond = False
            if self.adversarial:
                self.discriminator = MLP(inpt_dim=latent_dim+1, outp_dim=1, **adversarial_cfg.discriminator)
                if "double_discriminator" in self.adversarial:
                    self.discriminator2 = MLP(inpt_dim=x_dim+inpt_dim[1][0], outp_dim=1, **adversarial_cfg.discriminator2)
        else:
            self.encoder1 = MLP(inpt_dim=x_dim+inpt_dim[1][0], outp_dim=latent_dim, **encoder_cfg)
            if encoder_cfg2 is None:
                encoder_cfg2 = copy.deepcopy(encoder_cfg)
            if latent_dim2 is None:
                latent_dim2 = latent_dim
            self.encoder2 = MLP(inpt_dim=inpt_dim[1][0], outp_dim=latent_dim2, **encoder_cfg2)
            self.decoder_out_m = True
            self.decoder = MLP(inpt_dim=latent_dim+latent_dim2, outp_dim=x_dim+inpt_dim[1][0], **decoder_cfg)
            self.style_injection_cond = False
            if self.adversarial:
                self.discriminator = MLP(inpt_dim=latent_dim+1, outp_dim=1, **adversarial_cfg.discriminator)
                if "double_discriminator" in self.adversarial:
                    self.discriminator2 = MLP(inpt_dim=x_dim+inpt_dim[1][0], outp_dim=1, **adversarial_cfg.discriminator2)
        
        self.add_standardizing_layer = add_standardizing_layer
        if add_standardizing_layer:
            self.std_layer_x = IterativeNormLayer(x_dim)
            self.std_layer_ctxt = IterativeNormLayer(context_dim)
        
        self.var_group_list = var_group_list
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
        self.dis_steps_per_gen = 0

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

    def encode_content(self, x_inp, m_pair, mask=None):
        if not self.use_m_encodig:
            en = self.encoder1(x_inp, mask=mask)
        if self.style_injection_cond:
            if self.second_input_mask:
                en = self.encoder1(x_inp, mask=mask, ctxt=m_pair)
            else:
                en = self.encoder1(x_inp, ctxt=m_pair)
        else:
            en = self.encoder1(torch.cat([x_inp, m_pair], dim=1))
        
        if self.latent_norm_enc1:
            return normalize(en)
        else:
            return en
    
    def encode_style(self, m):
        en = self.encoder2(m)
        if self.latent_norm_enc2:
            return normalize(en)
        else:
            return en

    def decode(self, content, style, mask=None):
        if self.style_injection_cond:
            if self.second_input_mask:
                return self.decoder(content, mask=mask, ctxt=style)
            else:
                return self.decoder(content, ctxt=style)
        else:
            return self.decoder(torch.cat([content, style], dim=1))

    def disc_lat(self, e1, e2):
        if self.style_injection_cond:
            return self.discriminator(e1, ctxt=e2)
        else:
            return self.discriminator(torch.cat([e1, e2], dim=1))

    def disc_reco(self, w1, w2):
        if self.style_injection_cond:
            return self.discriminator2(w1, ctxt=w2)
        else:
            return self.discriminator2(torch.cat([w1, w2], dim=1))

    def _shared_step(self, sample: tuple, _batch_index = None, step_type="none") -> torch.Tensor:
        self.log(f"{step_type}_debug/global_step", self.global_step)
        batch_size=sample[0].shape[0]
        if self.second_input_mask:
            x_inp, mask, m_pair, m_add = sample
        else:
            x_inp, m_pair, m_add = sample
            mask = None
        
        #Make sure the inputs are in the right shape
        m_pair=m_pair.reshape([x_inp.shape[0], -1])
        m_add=m_add.reshape([x_inp.shape[0], -1])
        
        if self.add_standardizing_layer:
            # if step_type == "train":
            #     self.std_layer_x.update(x_inp, mask=mask)
            #     self.std_layer_ctxt.update(m_pair)
            x_inp = self.std_layer_x(x_inp, mask=mask)
            m_pair = self.std_layer_ctxt(m_pair)
            m_add = self.std_layer_ctxt(m_add)
        
        content = self.encode_content(x_inp, m_pair, mask=mask)
        style = self.encode_style(m_pair)
        
        recon = self.decode(content, style)
        
        # Reverse pass
        rpm = torch.randperm(batch_size)
        if self.reverse_pass_mode == "additional_input":
            style_p = self.encode_style(m_add)[rpm]
        else:
            style_p = style[rpm]

        recon_p = self.decode(content, style_p)
        if self.decoder_out_m:
            x_n = recon_p[:, :x_inp.shape[1]]
            m_n = recon_p[:, x_inp.shape[1]:]
        else:
            x_n = recon_p
            if self.reverse_pass_mode == "additional_input":
                m_n = m_add[rpm]
            else:
                m_n = m_pair[rpm]

        content_n = self.encode_content(x_n, m_n, mask=mask)
        style_n = self.encode_style(m_n)

        #### Losses
        total_loss = 0

        # Reconstruction loss
        if self.use_m_encodig and self.decoder_out_m:
            loss_reco = mse_loss(recon, torch.cat([x_inp, m_pair], dim=1)).mean()
        else:
            loss_reco = mse_loss(recon, x_inp).mean()
        total_loss += loss_reco*self.loss_cfg.reco.w
        self.log(f"{step_type}/loss_reco", loss_reco)
        
        # Second derivative smoothness
        if self.loss_cfg.second_derivative_smoothness is not None:
            e2_p_pl = self.encode_style(m_add + self.loss_cfg.second_derivative_smoothness.step)[rpm]
            e2_p_mi = self.encode_style(m_add - self.loss_cfg.second_derivative_smoothness.step)[rpm]
            recon_p_pl = self.decode(content, e2_p_pl)
            recon_p_mi = self.decode(content, e2_p_mi)
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
            loss_back_vec = mse_loss(content, content_n).mean()
            self.log(f"{step_type}/loss_back_vec", loss_back_vec)
            if self.loss_cfg.consistency_x.w is not None:
                if isinstance(self.loss_cfg.consistency_x.w, float) or isinstance(self.loss_cfg.consistency_x.w, int):
                    total_loss += loss_back_vec*self.loss_cfg.consistency_x.w
                else:
                    total_loss += loss_back_vec*self.loss_cfg.consistency_x.w(self.global_step)

        # if self.loss_cfg.consistency_x_balanced is not None:
        #     loss_back_vec = mse_loss(content, content_n).mean()/
        #     self.log(f"{step_type}/loss_back_vec", loss_back_vec)
        #     if self.loss_cfg.consistency_x.w is not None:
        #         if isinstance(self.loss_cfg.consistency_x.w, float) or isinstance(self.loss_cfg.consistency_x.w, int):
        #             total_loss += loss_back_vec*self.loss_cfg.consistency_x.w
        #         else:
        #             total_loss += loss_back_vec*self.loss_cfg.consistency_x.w(self.global_step)

        if self.loss_cfg.consistency_cont is not None:
            loss_back_cont = mse_loss(style_p, style_n).mean()
            self.log(f"{step_type}/loss_back_cont", loss_back_cont)
            if self.loss_cfg.consistency_cont.w is not None:
                if isinstance(self.loss_cfg.consistency_cont.w, float) or isinstance(self.loss_cfg.consistency_cont.w, int):
                    total_loss += loss_back_cont*self.loss_cfg.consistency_cont.w
                else:
                    total_loss += loss_back_cont*self.loss_cfg.consistency_cont.w(self.global_step)

        # variance loss
        if self.loss_cfg.latent_variance_cfg is not None:
            std_e1 = torch.sqrt(content.var(dim=0) + 0.0001)
            #std_e2 = torch.sqrt(style.var(dim=0) + 0.0001)
            loss_latent_variance = torch.mean(torch.abs(1 - std_e1)**self.loss_cfg.latent_variance_cfg.pow)# + torch.mean(torch.square(1 - std_e2)**self.loss_cfg.latent_variance_cfg.pow) / 2
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
            loss_attractive = -cosine_similarity(content, style[torch.randperm(batch_size)]).mean()
            self.log(f"{step_type}/loss_attractive", loss_attractive)
            if self.loss_cfg.attractive.w is not None:
                if isinstance(self.loss_cfg.attractive.w, float) or isinstance(self.loss_cfg.attractive.w, int):
                    total_loss += loss_attractive*self.loss_cfg.attractive.w
                else:
                    total_loss += loss_attractive*self.loss_cfg.attractive.w(self.global_step)
        
        if self.loss_cfg.repulsive is not None:
            loss_repulsive = torch.abs(cosine_similarity(content, style)).mean()
            self.log(f"{step_type}/loss_repulsive", loss_repulsive)
            if self.loss_cfg.repulsive.w is not None:
                if isinstance(self.loss_cfg.repulsive.w, float) or isinstance(self.loss_cfg.repulsive.w, int):
                    total_loss += loss_repulsive*self.loss_cfg.repulsive.w
                else:
                    total_loss += loss_repulsive*self.loss_cfg.repulsive.w(self.global_step)

        # DisCO loss
        if self.loss_cfg.DisCO_loss_cfg is not None:
            if self.loss_cfg.DisCO_loss_cfg.mode == "e1_vs_e2":
                loss_disco = self.DisCO_loss(content, style)
            elif self.loss_cfg.DisCO_loss_cfg.mode == "e1_vs_w2":
                loss_disco = self.DisCO_loss(content, m_pair)
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
            loss_disco = self.pearson_loss(content, style)
            self.log(f"{step_type}/pearson_loss", loss_disco)
            if self.loss_cfg.pearson_loss_cfg.w is not None:
                if isinstance(self.loss_cfg.pearson_loss_cfg.w, float) or isinstance(self.loss_cfg.pearson_loss_cfg.w, int):
                        total_loss += loss_disco*self.loss_cfg.pearson_loss_cfg.w
                else:
                    total_loss += loss_disco*self.loss_cfg.pearson_loss_cfg.w(self.global_step)

        # CLIP loss
        if self.loss_cfg.clip_loss_cfg is not None:
            loss_clip = self.clip_loss(content, style).mean()
            self.log(f"{step_type}/clip_loss", loss_clip)
            if self.loss_cfg.clip_loss_cfg.w is not None:
                total_loss += loss_clip*self.loss_cfg.clip_loss_cfg.w

        # VIC loss
        if self.loss_cfg.vic_reg_cfg is not None:
            loss, repr_loss, std_loss, cov_loss = self.VICloss(content, style)
            self.log(f"{step_type}/VIC_repr_loss", repr_loss)
            self.log(f"{step_type}/VIC_std_loss", std_loss)
            self.log(f"{step_type}/VIC_cov_loss", cov_loss)
            total_loss += loss

        # Log the total loss
        self.log(f"{step_type}/total_loss", total_loss)
        if self.adversarial:
            return total_loss, content, style, x_inp, m_pair
        else:
            return total_loss

    def adversarial_loss(self, y_hat, y):
        if self.adversarial_cfg.loss_function=="binary_cross_entropy":
            return F.binary_cross_entropy(y_hat, y.reshape((-1, 1))).mean()
        elif self.adversarial_cfg.loss_function=="mse":
            return mse_loss(y_hat, y.reshape((-1, 1))).mean()
        elif self.adversarial_cfg.loss_function=="hinge":
            return F.hinge_embedding_loss((y_hat-0.5)*2, (y.reshape((-1, 1))-0.5)*2).mean()

    def training_step(self, sample: tuple, batch_idx: int) -> torch.Tensor:

        if self.adversarial=="latent_assymetric":
            optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
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
                self.zero_grad()
                self.manual_backward(d_loss) #retain_graph=True
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
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
                self.zero_grad()
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                self.untoggle_optimizer(optimizer_g)
        elif self.adversarial=="latent_gaussian":
            optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
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
                self.zero_grad()
                self.manual_backward(d_loss, retain_graph=True)
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or self.global_step%self.adversarial_cfg.every_n_steps_g==0:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    g_loss = - self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, torch.normal(0, 1, size=e1.shape).to(e1.device)], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight
                else:
                    total_loss2 = total_loss
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.zero_grad()
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                self.untoggle_optimizer(optimizer_g)
        elif self.adversarial=="to0.5": 
            optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
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
                self.zero_grad()
                self.manual_backward(d_loss, retain_graph=True)
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or self.global_step%self.adversarial_cfg.every_n_steps_g==0:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    labels = torch.cat([torch.ones(batch_size*2)*0.5]).type_as(w2_perm)
                    g_loss = self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    self.log("g_loss", g_loss, prog_bar=True)
                    total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight
                else:
                    total_loss2 = total_loss
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.zero_grad()
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                self.untoggle_optimizer(optimizer_g)
        elif self.adversarial=="oposite_labels": 
            optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            batch_size=sample[0].shape[0]
            rpm = torch.randperm(batch_size)
            w2_perm = w2.clone()
            w2_perm = w2_perm[rpm]
            e1_copy = e1.clone()
            # train discriminator
            # Measure discriminator's ability to classify real from generated samples
            if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).type_as(w2_perm)
                d_loss = self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                self.toggle_optimizer(optimizer_d)
                self.log("d_loss", d_loss, prog_bar=True)
                self.zero_grad()
                self.manual_backward(d_loss, retain_graph=True)
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or self.global_step%self.adversarial_cfg.every_n_steps_g==0:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    labels = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)]).type_as(w2_perm)
                    g_loss = self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    self.log("g_loss", g_loss, prog_bar=True)
                    total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight
                else:
                    total_loss2 = total_loss
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.zero_grad()
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                self.untoggle_optimizer(optimizer_g)
        elif self.adversarial=="3optim_normal": 
            optimizer_e, optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            if self.current_epoch<self.adversarial_cfg.warmup or self.global_step%self.adversarial_cfg.every_n_steps_g==0:
                self.toggle_optimizer(optimizer_g)
                self.zero_grad()
                self.manual_backward(total_loss, retain_graph=True)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                self.untoggle_optimizer(optimizer_g)  
            
            e1 = self.encode_w1(sample[0])
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
                self.log("d_loss", d_loss, prog_bar=True)
                self.toggle_optimizer(optimizer_d)
                self.zero_grad()
                self.manual_backward(d_loss, retain_graph=True)
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or self.global_step%self.adversarial_cfg.every_n_steps_g==0:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    labels = torch.cat([torch.ones(batch_size*2)]).type_as(w2_perm)
                    g_loss = - self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    self.log("g_loss", g_loss)
                    self.toggle_optimizer(optimizer_e)
                    self.zero_grad()
                    self.manual_backward(g_loss*self.adversarial_cfg.g_loss_weight)
                    self.clip_gradients(optimizer_e, gradient_clip_val=self.gradient_clip_val)
                    optimizer_e.step()
                    self.untoggle_optimizer(optimizer_e) 
        elif self.adversarial=="opt_grad_clean": 
            optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
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
        elif "double_discriminator" in self.adversarial: 
            optimizer_g, optimizer_d, optimizer_d2 = self.optimizers()
            # adversarial loss is binary cross-entropy
            total_loss, e1, e2, w1, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            batch_size=sample[0].shape[0]
            rpm = torch.randperm(batch_size)
            w2_perm = w2.clone()
            w2_perm = w2_perm[rpm]
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).type_as(w2_perm)
            e1_copy = e1.clone()
            generated = self.decode(e1, e2[rpm])
            if self.adversarial_cfg.loss_function=="binary_cross_entropy":
                threshold=np.log(2)
            elif self.adversarial_cfg.loss_function=="mse":
                threshold=0.25
            # train discriminator
            # Measure discriminator's ability to classify encoded samples with correct mass and encoded samples with incorrect mass
            allow_gen_train = True
            if self.current_epoch>=self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                # Train discriminator for latent space
                d_loss = self.adversarial_loss(self.disc_lat(torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)), labels)
                self.toggle_optimizer(optimizer_d)
                self.log("d_loss", d_loss, prog_bar=True)
                self.zero_grad()
                self.manual_backward(d_loss, retain_graph=True)
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)
                
                # Train discriminator for reconstruction/transport
                d_loss_gen= self.adversarial_loss(self.disc_reco(torch.cat([w1, generated], dim=0), torch.cat([w2, w2_perm], dim=0)),  labels)
                self.toggle_optimizer(optimizer_d2)
                self.log("d_loss_gen", d_loss_gen, prog_bar=True)
                self.zero_grad()
                self.manual_backward(d_loss_gen, retain_graph=True)
                self.clip_gradients(optimizer_d2, gradient_clip_val=self.gradient_clip_val)
                optimizer_d2.step()
                self.untoggle_optimizer(optimizer_d2)
                self.dis_steps_per_gen+=1
                
                # Pause genrator training if discriminator is too weak
                if self.adversarial=="double_discriminator_priority":
                    if d_loss>threshold or d_loss_gen>threshold:
                        allow_gen_train = False
                if self.adversarial=="double_discriminator_priority_balancing":
                    if d_loss<0.69 or d_loss_gen<0.69:
                        self.adversarial_cfg.every_n_steps_g = max(self.adversarial_cfg.every_n_steps_g-1, 1)
                    elif d_loss>threshold or d_loss_gen>threshold:
                        self.adversarial_cfg.every_n_steps_g = min(self.adversarial_cfg.every_n_steps_g+1, 5)
                    
                    if d_loss>threshold or d_loss_gen>threshold:
                        allow_gen_train = False
 
                        
                if self.dis_steps_per_gen<self.adversarial_cfg.every_n_steps_g:
                    allow_gen_train = False
                        
            if self.current_epoch>self.afterglow_epoch:
                allow_gen_train = False

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or allow_gen_train:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    g_loss = - self.adversarial_loss(self.disc_lat(torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)), labels)
                    g_loss_gen = - self.adversarial_loss(self.disc_reco(torch.cat([w1, generated], dim=0), torch.cat([w2, w2_perm], dim=0)),  labels)                        
                    if isinstance(self.adversarial_cfg.g_loss_weight, float) or isinstance(self.adversarial_cfg.g_loss_weight, int):
                        g_loss_weight = self.adversarial_cfg.g_loss_weight
                    else:
                        g_loss_weight = self.adversarial_cfg.g_loss_weight(self.global_step)
                        self.log("g_loss_weight", g_loss_weight)
                    if isinstance(self.adversarial_cfg.g_loss_gen_weight, float) or isinstance(self.adversarial_cfg.g_loss_gen_weight, int):
                        g_loss_gen_weight = self.adversarial_cfg.g_loss_gen_weight
                    else:
                        g_loss_gen_weight = self.adversarial_cfg.g_loss_gen_weight(self.global_step)
                        self.log("g_loss_gen_weight", g_loss_gen_weight)
                    total_loss2 = total_loss + g_loss*g_loss_weight + g_loss_gen*g_loss_gen_weight
                else:
                    total_loss2 = total_loss
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.zero_grad()
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                self.untoggle_optimizer(optimizer_g)
                self.log("dis_steps_per_gen", self.dis_steps_per_gen)
                self.dis_steps_per_gen = 0

        elif self.adversarial=="discriminator_priority": 
            optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            batch_size=sample[0].shape[0]
            rpm = torch.randperm(batch_size)
            w2_perm = w2.clone()
            w2_perm = w2_perm[rpm]
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).type_as(w2_perm)
            e1_copy = e1.clone()
            if self.adversarial_cfg.loss_function=="binary_cross_entropy":
                threshold=np.log(2)
            elif self.adversarial_cfg.loss_function=="mse":
                threshold=0.25
            # train discriminator
            # Measure discriminator's ability to classify real from generated samples
            if self.current_epoch>=self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                d_loss = self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                self.toggle_optimizer(optimizer_d)
                self.log("d_loss", d_loss, prog_bar=True)
                self.zero_grad()
                self.manual_backward(d_loss, retain_graph=True)
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)
                self.dis_steps_per_gen+=1

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or d_loss<threshold:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    g_loss = - self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight
                else:
                    total_loss2 = total_loss
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.zero_grad()
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                self.untoggle_optimizer(optimizer_g)
                self.log("dis_steps_per_gen", self.dis_steps_per_gen)
                self.dis_steps_per_gen = 0
        elif self.adversarial=="default": 
            optimizer_g, optimizer_d = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
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
                self.zero_grad()
                self.manual_backward(d_loss, retain_graph=True)
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or self.global_step%self.adversarial_cfg.every_n_steps_g==0:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    g_loss = - self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight
                else:
                    total_loss2 = total_loss
                self.log("total_loss2", total_loss2, prog_bar=True)
                self.zero_grad()
                self.manual_backward(total_loss2)
                self.clip_gradients(optimizer_g, gradient_clip_val=self.gradient_clip_val)
                optimizer_g.step()
                self.untoggle_optimizer(optimizer_g)
        elif self.adversarial:
            assert False, "Adversarial mode not implemented"
        else:	
            total_loss = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            return total_loss

    def _draw_event_transport_trajectories(self, w1_, m_pair_, var, var_name, masses=np.linspace(-2.5, 2.5, 126), max_traj=20):
        w1 = copy.deepcopy(w1_)[:max_traj]
        m_pair = m_pair_[:max_traj]
        content = self.encode_content(w1, m_pair)
        recons = []
        if self.adversarial:
            zs = []
        for m in masses:
            w2 = torch.tensor(m).unsqueeze(0).expand(w1.shape[0], 1).float().to(w1.device)
            style = self.encode_style(w2)
            recon = self.decode(content, style)
            recons.append(recon)
            if self.adversarial:
                zs.append(self.disc_lat(content, style))
        if self.adversarial:
            vmin = min([float(z[:max_traj].min().cpu().detach().numpy()) for z in zs])
            vmax = max([float(z[:max_traj].max().cpu().detach().numpy()) for z in zs])
        plt.figure()
        if max_traj is None:
            max_traj = x.shape[0]
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
            plt.scatter(to_np(m_pair)[:max_traj], to_np(w1[:, var])[:max_traj],  marker="x", label="originals", c="green")
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

    def _draw_event_transport_trajectories_2nd_der(self, w1_, m_pair_, var, var_name, masses=None, max_traj=20):
        w1 = copy.deepcopy(w1_)[:max_traj]
        m_pair_ = m_pair_[:max_traj]
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
            content = self.encode_content(w1, m_pair)
            style = self.encode_style(w2)
            recon = self.decode(content, style)
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
            if self.second_input_mask:
                w2 = sample[2]
            else:
                w2 = sample[1]
            if self.add_standardizing_layer:
                w1 = self.std_layer_x(w1)
                w2 = self.std_layer_ctxt(w2)
            for var in range(w1.shape[1]):
                image = wandb.Image(self._draw_event_transport_trajectories(w1, w2, var=var, var_name=self.var_group_list[0][var], max_traj=20))
                if wandb.run is not None:
                    wandb.run.log({f"valid_images/transport_{self.var_group_list[0][var]}": image})
                # image = wandb.Image(self._draw_event_transport_trajectories_2nd_der(sample[0], sample[1], var=var, var_name=self.var_group_list[0][var], max_traj=20))
                # if wandb.run is not None:
                #     wandb.run.log({f"valid_images/transport_2nd_der_{self.var_group_list[0][var]}": image})
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
        if self.adversarial=="3optim_normal":
            enc2_params =  list(self.encoder2.parameters()) if hasattr(self.encoder2, "parameters") else []
            enc_dec_params = list(self.encoder1.parameters()) + enc2_params + list(self.decoder.parameters())
            opt_e = self.hparams.optimizer(params=self.encoder1.parameters())
            opt_g = self.hparams.optimizer(params=enc_dec_params)
            opt_d = self.hparams.optimizer(params=self.discriminator.parameters())
            if getattr(self.adversarial_cfg, "scheduler", None) is None:
                return [opt_e, opt_g, opt_d], []
            elif self.adversarial_cfg.scheduler == "same_given":
                sched_g = self.hparams.scheduler.scheduler(opt_g)
                sched_d = self.hparams.scheduler.scheduler(opt_d)
                sched_e = self.hparams.scheduler.scheduler(opt_e)
                return [opt_e, opt_g, opt_d], [sched_e, sched_g, sched_d]
            else: 
                sched_g = self.adversarial_cfg.scheduler.scheduler_g(opt_g)
                sched_d = self.adversarial_cfg.scheduler.scheduler_d(opt_d)
                sched_e = self.adversarial_cfg.scheduler.scheduler_e(opt_e)
                return [opt_e, opt_g, opt_d], [sched_e, sched_g, sched_d]            
        elif "double_discriminator" in self.adversarial:
            if hasattr(self, "encoder2"):
                enc2_params =  list(self.encoder2.parameters()) if hasattr(self.encoder2, "parameters") else []
            else:
                enc2_params = []
            enc_dec_params = list(self.encoder1.parameters()) + enc2_params + list(self.decoder.parameters())
            opt_g = self.hparams.optimizer(params=enc_dec_params)
            opt_d = self.hparams.optimizer(params=self.discriminator.parameters())
            opt_d2 = self.hparams.optimizer(params=self.discriminator2.parameters())
            if getattr(self.adversarial_cfg, "scheduler", None) is None:
                return [opt_g, opt_d, opt_d2], []
            elif self.adversarial_cfg.scheduler == "same_given":
                sched_g = self.hparams.scheduler.scheduler(opt_g)
                sched_d = self.hparams.scheduler.scheduler(opt_d)
                sched_d2 = self.hparams.scheduler.scheduler(opt_d2)
                return [opt_g, opt_d, opt_d2], [sched_g, sched_d, sched_d2]
            else: 
                sched_g = self.adversarial_cfg.scheduler.scheduler_g(opt_g)
                sched_d = self.adversarial_cfg.scheduler.scheduler_d(opt_d)
                sched_d2 = self.adversarial_cfg.scheduler.scheduler_d2(opt_d2)
                return [opt_g, opt_d, opt_d2], [sched_g, sched_d, sched_d2]
        elif self.adversarial:	
            enc2_params =  list(self.encoder2.parameters()) if hasattr(self.encoder2, "parameters") else []
            enc_dec_params = list(self.encoder1.parameters()) + enc2_params + list(self.decoder.parameters())
            opt_g = self.hparams.optimizer(params=enc_dec_params)
            opt_d = self.hparams.optimizer(params=self.discriminator.parameters())
            if getattr(self.adversarial_cfg, "scheduler", None) is None:
                return [opt_g, opt_d], []
            elif self.adversarial_cfg.scheduler == "same_given":
                sched_g = self.hparams.scheduler.scheduler(opt_g)
                sched_d = self.hparams.scheduler.scheduler(opt_d)
                return [opt_g, opt_d], [sched_g, sched_d]
            else: 
                sched_g = self.adversarial_cfg.scheduler.scheduler_g(opt_g)
                sched_d = self.adversarial_cfg.scheduler.scheduler_d(opt_d)
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
        if self.second_input_mask:
            x_inp, mask, y_pair, y_new = sample
        else:
            x_inp, y_pair, y_new = sample
            mask = None
        
        if self.add_standardizing_layer:
            x_inp = self.std_layer_x(x_inp)
            content = self.std_layer_ctxt(y_pair)
        
        content = self.encode_content(x_inp, y_pair, mask=mask)
        style = self.encode_style(y_new)
        recon = self.decode(content, style)
        
        if self.add_standardizing_layer:
            recon = self.std_layer_x.reverse(recon)
        
        return recon

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if self.second_input_mask:
            context = batch[3].reshape([len(batch[0]), -1])
        else:
            context = batch[2].reshape([len(batch[0]), -1])
        sample = self.generate(batch).squeeze(1)
        sample = sample.reshape(-1, sample.shape[-1])
        if self.var_group_list is not None:
            result = {var_name: column.reshape(-1, 1) for var_name, column in zip(self.var_group_list[0], sample.T)}
            result.update({var_name: column.reshape(-1, 1) for var_name, column in zip(self.var_group_list[1], context.T)})                
            return result
        else:
            result = sample
            return result

