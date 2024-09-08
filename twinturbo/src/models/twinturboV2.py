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
import copy
from mattstools.mattstools.simple_transformers import TransformerEncoder, FullEncoder

class TwinTURBOv2(LightningModule):
    
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
        network_type = "MLP",
        encoder_mlp_config2: Mapping = None,
        discriminator_latent_cfg = None,
        latent_dim2: int = None, 
        var_group_list: list = None,
        loss_cfg: Mapping = None,
        use_m_encodig = True,
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
            if "double_discriminator" in self.adversarial:
                self.discriminator2 = MLP(inpt_dim=inpt_dim[0][0]+inpt_dim[1][0], outp_dim=1, **adversarial_cfg.discriminator2)
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
        
        
        if network_type == "Transformer_conditional":
            self.encoder = TransformerEncoder(**encoder_mlp_config)
            self.decoder = FullEncoder(**decoder_mlp_config)
            if discriminator_latent_cfg is not None:
                self.discriminator = TransformerEncoder(inpt_dim=latent_dim, outp_dim=1, **discriminator_latent_cfg)
            if discriminator_latent_cfg is not None:
                self.discriminator = TransformerEncoder(inpt_dim=latent_dim, outp_dim=1, **discriminator_latent_cfg)
            self.style_injection = "conditional"
        elif network_type == "MLP_cont":
            self.decoder_out_m = False
            self.encoder = MLP(inpt_dim=inpt_dim[0][0], ctxt_dim=inpt_dim[1][0], outp_dim=latent_dim, **encoder_mlp_config)
            self.decoder = MLP(inpt_dim=latent_dim, ctxt_dim=inpt_dim[1][0], outp_dim=inpt_dim[0][0], **decoder_mlp_config)
            self.style_injection = "conditional"
        elif network_type == "MLP_no_m_encoding":
            self.encoder1 = MLP(inpt_dim=inpt_dim[0][0]+inpt_dim[1][0], outp_dim=latent_dim, **encoder_mlp_config)
            latent_dim2 = 1
            self.encoder2 = lambda x: x
            self.decoder_out_m = False
            self.decoder = MLP(inpt_dim=latent_dim+latent_dim2, outp_dim=inpt_dim[0][0], **decoder_mlp_config) 
            self.latent_norm_enc2 = False
            self.style_injection = "concat"
        else:
            self.encoder1 = MLP(inpt_dim=inpt_dim[0][0]+inpt_dim[1][0], outp_dim=latent_dim, **encoder_mlp_config)
            if encoder_mlp_config2 is None:
                encoder_mlp_config2 = copy.deepcopy(encoder_mlp_config)
            if latent_dim2 is None:
                latent_dim2 = latent_dim
            self.encoder2 = MLP(inpt_dim=inpt_dim[1][0], outp_dim=latent_dim2, **encoder_mlp_config2)
            self.decoder_out_m = True
            self.decoder = MLP(inpt_dim=latent_dim+latent_dim2, outp_dim=inpt_dim[0][0]+inpt_dim[1][0], **decoder_mlp_config)
            self.style_injection = "concat"
            
        self.var_group_list = var_group_list
        self.reverse_pass_mode = reverse_pass_mode
        self.input_noise_cfg = input_noise_cfg

        # For more stable checks in the shared step
        expected_attrs = ["reco", 
                        "consistency_x", 
                        "consistency_cont",
                        "second_derivative_smoothness"]
        for attr in list(self.loss_cfg.keys()):
            if attr in expected_attrs:
                setattr(self.loss_cfg, attr, getattr(self.loss_cfg, attr, None))
            else:
                assert False, f"Unexpected loss config attribute: {attr}"
        for attr in expected_attrs:
            if not hasattr(self.loss_cfg, attr):
                setattr(self.loss_cfg, attr, None)

        self.dis_steps_per_gen = 0

    def encode_content(self, x_inp, m_pair):
        if not self.use_m_encodig:
            en = self.encoder1(x_inp)
        elif self.style_injection == "conditional":
            en = self.encoder(x_inp, m_pair)
        elif self.style_injection == "concat":
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

    def decode(self, content, style):
        if self.network_type == "MLP_cont":
            en = self.decoder(content, style)
        else:
            return self.decoder(torch.cat([content, style], dim=1))

    def _shared_step(self, sample: tuple, _batch_index = None, step_type="none") -> torch.Tensor:
        self.log(f"{step_type}_debug/global_step", self.global_step)
        batch_size=sample[0].shape[0]
        x_inp = sample[0]
        m_pair = sample[1]
        
        content = self.encode_content(x_inp, m_pair)
        style = self.encode_style(m_pair)
        
        recon = self.decode(content, style)
        
        # Reverse pass
        rpm = torch.randperm(batch_size)
        if self.reverse_pass_mode == "additional_input":
            style_p = self.encode_style(sample[2])[rpm]
        else:
            style_p = style[rpm]

        recon_p = self.decode(content, style_p)
        if self.decoder_out_m:
            x_n = recon_p[:, :x_inp.shape[1]]
            m_n = recon_p[:, x_inp.shape[1]:]
        else:
            x_n = recon_p
            if self.reverse_pass_mode == "additional_input":
                m_n = sample[2][rpm]
            else:
                m_n = sample[1][rpm]

        content_n = self.encode_content(x_n, m_n)
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
            e2_p_pl = self.encode_style(sample[2] + self.loss_cfg.second_derivative_smoothness.step)[rpm]
            e2_p_mi = self.encode_style(sample[2] - self.loss_cfg.second_derivative_smoothness.step)[rpm]
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
        
        if self.loss_cfg.consistency_cont is not None:
            loss_back_cont = mse_loss(style_p, style_n).mean()
            self.log(f"{step_type}/loss_back_cont", loss_back_cont)
            if self.loss_cfg.consistency_cont.w is not None:
                if isinstance(self.loss_cfg.consistency_cont.w, float) or isinstance(self.loss_cfg.consistency_cont.w, int):
                    total_loss += loss_back_cont*self.loss_cfg.consistency_cont.w
                else:
                    total_loss += loss_back_cont*self.loss_cfg.consistency_cont.w(self.global_step)

        # Log the total loss
        self.log(f"{step_type}/total_loss", total_loss)
        if self.adversarial:
            return total_loss, content, style, m_pair
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
        elif self.adversarial=="double_discriminator_default": 
            optimizer_g, optimizer_d, optimizer_d2 = self.optimizers()
            # adversarial loss is binary cross-entropy
            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
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
            if self.current_epoch>=self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                d_loss = self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                self.toggle_optimizer(optimizer_d)
                self.log("d_loss", d_loss, prog_bar=True)
                self.zero_grad()
                self.manual_backward(d_loss, retain_graph=True)
                self.clip_gradients(optimizer_d, gradient_clip_val=self.gradient_clip_val)
                optimizer_d.step()
                self.untoggle_optimizer(optimizer_d)
                #self.dis_steps_per_gen+1
                
            if self.current_epoch>=self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                d_loss_gen = self.adversarial_loss(self.discriminator2(torch.cat([torch.cat([sample[0], sample[1]], dim=1), generated], dim=0)), labels)
                self.toggle_optimizer(optimizer_d2)
                self.log("d_loss_gen", d_loss_gen, prog_bar=True)
                self.zero_grad()
                self.manual_backward(d_loss_gen, retain_graph=True)
                self.clip_gradients(optimizer_d2, gradient_clip_val=self.gradient_clip_val)
                optimizer_d2.step()
                self.untoggle_optimizer(optimizer_d2)
                self.dis_steps_per_gen+=1

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or self.dis_steps_per_gen%self.adversarial_cfg.every_n_steps_g==0:
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    g_loss = - self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    g_loss_gen = - self.adversarial_loss(self.discriminator2(torch.cat([torch.cat([sample[0], sample[1]], dim=1), generated], dim=0)), labels)
                    total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight + g_loss_gen*self.adversarial_cfg.g_loss_weight
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
        elif self.adversarial=="double_discriminator_priority": 
            optimizer_g, optimizer_d, optimizer_d2 = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
            batch_size=sample[0].shape[0]
            rpm = torch.randperm(batch_size)
            w2_perm = w2.clone()
            w2_perm = w2_perm[rpm]
            labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).type_as(w2_perm)
            e1_copy = e1.clone()
            generated = self.decode(e1, e2[rpm])
            if not self.decoder_out_m:
                generated = torch.cat([generated, w2_perm], dim=1)
            if self.adversarial_cfg.loss_function=="binary_cross_entropy":
                threshold=np.log(2)
            elif self.adversarial_cfg.loss_function=="mse":
                threshold=0.25
            # train discriminator
            # Measure discriminator's ability to classify encoded samples with correct mass and encoded samples with incorrect mass
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
                
            if self.current_epoch>=self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                d_loss_gen = self.adversarial_loss(self.discriminator2(torch.cat([torch.cat([sample[0], sample[1]], dim=1), generated], dim=0)), labels)
                self.toggle_optimizer(optimizer_d2)
                self.log("d_loss_gen", d_loss_gen, prog_bar=True)
                self.zero_grad()
                self.manual_backward(d_loss_gen, retain_graph=True)
                self.clip_gradients(optimizer_d2, gradient_clip_val=self.gradient_clip_val)
                optimizer_d2.step()
                self.untoggle_optimizer(optimizer_d2)
                self.dis_steps_per_gen+=1

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or (d_loss<threshold and d_loss_gen<threshold):
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    g_loss = - self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    g_loss_gen = - self.adversarial_loss(self.discriminator2(torch.cat([torch.cat([sample[0], sample[1]], dim=1), generated], dim=0)), labels)
                    total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight + g_loss_gen*self.adversarial_cfg.g_loss_weight
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
        elif self.adversarial=="double_discriminator_priority_same_oder": 
            optimizer_g, optimizer_d, optimizer_d2 = self.optimizers()
            # adversarial loss is binary cross-entropy

            total_loss, e1, e2, w2 = self._shared_step(sample, step_type="train", _batch_index=batch_idx)
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
                
            if self.current_epoch>=self.adversarial_cfg.warmup or self.adversarial_cfg.train_dis_in_warmup:
                d_loss_gen = self.adversarial_loss(self.discriminator2(torch.cat([torch.cat([sample[0], sample[1]], dim=1), generated], dim=0)), labels)
                self.toggle_optimizer(optimizer_d2)
                self.log("d_loss_gen", d_loss_gen, prog_bar=True)
                self.zero_grad()
                self.manual_backward(d_loss_gen, retain_graph=True)
                self.clip_gradients(optimizer_d2, gradient_clip_val=self.gradient_clip_val)
                optimizer_d2.step()
                self.untoggle_optimizer(optimizer_d2)
                self.dis_steps_per_gen+=1

            # Train generator
            if self.current_epoch<self.adversarial_cfg.warmup or (d_loss<threshold and d_loss_gen<threshold):
                if self.current_epoch>self.adversarial_cfg.warmup or self.adversarial_cfg.g_loss_weight_in_warmup:
                    g_loss = - self.adversarial_loss(self.discriminator(torch.cat([torch.cat([e1, e1_copy], dim=0), torch.cat([w2, w2_perm], dim=0)], dim=1)), labels)
                    g_loss_gen = - self.adversarial_loss(self.discriminator2(torch.cat([torch.cat([sample[0], sample[1]], dim=1), generated], dim=0)), labels)
                    total_loss2 = total_loss + g_loss*self.adversarial_cfg.g_loss_weight + g_loss_gen*self.adversarial_cfg.g_loss_weight
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

    def validation_step (self, sample: tuple, batch_idx: int) -> torch.Tensor:
        total_loss = self._shared_step(sample, step_type="valid", _batch_index=batch_idx)
        if batch_idx == 0 and self.valid_plots:
            w1 = sample[0]
            for var in range(w1.shape[1]):
                image = wandb.Image(self._draw_event_transport_trajectories(sample[0], sample[1], var=var, var_name=self.var_group_list[0][var], max_traj=20))
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
        elif self.adversarial=="double_discriminator_priority":
            enc2_params =  list(self.encoder2.parameters()) if hasattr(self.encoder2, "parameters") else []
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
    
    def generate(self, sample: tuple) -> torch.Tensor:
        x_inp, y_pair, y_new = sample
        content = self.encode_content(x_inp, y_pair)
        style = self.encode_style(y_new)
        recon = self.decode(content, style)
        return recon

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        if len(batch)==3:
            context = batch[2]
            sample = self.generate(batch).squeeze(1)
            sample = sample.reshape(-1, sample.shape[-1])
            result = {var_name: column.reshape(-1, 1) for var_name, column in zip(self.var_group_list[0], sample.T)}
            result.update({var_name: column.reshape(-1, 1) for var_name, column in zip(self.var_group_list[1], context.T)})
            return result

