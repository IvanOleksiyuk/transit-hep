import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from typing import Union
import logging
import hydra
from hydra.core.global_hydra import GlobalHydra

from pathlib import Path
import os
from twinturbo.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config

import pandas as pd
import twinturbo.src.utils.plotting as pltt

import matplotlib.pyplot as plt
import numpy as np
from twinturbo.src.data.lhco_simple import LHCOInMemoryDataset
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import torch as T
from torch.nn.functional import normalize, mse_loss, cosine_similarity
log = logging.getLogger(__name__)

def to_np(inpt: Union[T.Tensor, tuple]) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a pytorch
    tensor to numpy array.

    - Includes gradient deletion, and device migration
    """
    if isinstance(inpt, (tuple, list)):
        return type(inpt)(to_np(x) for x in inpt)
    if inpt.dtype == T.bfloat16:  # Numpy conversions don't support bfloat16s
        inpt = inpt.half()
    return inpt.detach().cpu().numpy()


@rank_zero_only
def reload_original_config(cfg: OmegaConf, get_best: bool = False) -> OmegaConf:
    """Replaces the cfg with the one stored at the checkpoint location.

    Will also set the chkpt_dir to the latest version of the last or
    best checkpoint
    """

    # Load the original config found in the the file directory
    orig_cfg = OmegaConf.load(Path(cfg.paths.full_path+"/full_config.yaml"))

    # Get the latest updated checkpoint with the prefix last or best
    flag = "best" if get_best else "last"
    orig_cfg.ckpt_path = str(
        sorted(Path(cfg.paths.full_path).glob(f"checkpoints/{flag}*.ckpt"), key=os.path.getmtime)[-1]
    )

    # Set the wandb logger to attempt to resume the job
    if hasattr(orig_cfg, "loggers"):
        if hasattr(orig_cfg.loggers, "wandb"):
            orig_cfg.loggers.wandb.resume = True

    return orig_cfg

def main(cfg):
	print("Starting evaluation")
	# Get two dataframes to compare	
	original_data = LHCOInMemoryDataset(**cfg.export_template.data.train_data_conf.dataset1_cfg, do_group_split = False, do_plotting = False).data[0]
	bkg_data_sig_region = LHCOInMemoryDataset(**cfg.train_cwola.data.train_data_conf, do_group_split = False, do_plotting = False).data[0]
	template_data = pd.read_hdf(cfg.export_template.paths.full_path + "/outputs/template_sample.h5")
	
	#bkg_data_sig_region["m_jj"]=(bkg_data_sig_region["m_jj"]-np.mean(bkg_data_sig_region["m_jj"]))/np.std(bkg_data_sig_region["m_jj"])
	#template_data["m_jj"]=(template_data["m_jj"]-np.mean(template_data["m_jj"]))/np.std(template_data["m_jj"])
	
	variables = ["m_n", "m_jj"] #"m_j1", "m_j2", "del_m", "tau21_j1", "tau32_j1", "tau21_j2", "tau32_j2",
	print(len(bkg_data_sig_region.to_numpy()))
	print(len(template_data.to_numpy()))

	pltt.plot_feature_spread(
    bkg_data_sig_region[variables].to_numpy(),
    template_data[variables].to_numpy(),
    original_data = original_data[variables].to_numpy(),
    feature_nms = variables,
    save_dir=Path(cfg.general.run_dir),
    plot_mode="diagnose",)
 	
	evaluate_model(cfg)

def plot_matrix(matrix, title, vmin=-1, vmax=1, abs=False):
	fig, ax = plt.subplots(figsize=(8, 8))
	if abs:
		im = ax.imshow(np.abs(matrix), vmin=0, vmax=vmax, cmap="Blues")
	else:
		im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="RdBu")
	ax.set_title(title)
	fig.colorbar(im, ax=ax)
	return fig, ax

def evaluate_model(cfg):
	if GlobalHydra().is_initialized():
		GlobalHydra().clear()
	hydra.initialize(version_base=None, config_path= "../conf")
	cfg = hydra.compose(config_name="twinturbo.yaml")
	log.info("Loading run information")
	cfg = cfg.export_template
	print(cfg.paths.full_path)
	orig_cfg = reload_original_config(cfg, get_best=True)
	cfg = orig_cfg

	log.info("Loading best checkpoint")
	device = "cuda" if T.cuda.is_available() else "cpu"
	model_class = hydra.utils.get_class(orig_cfg.model._target_)
	model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location=device)

	log.info("Instantiating original trainer")
	trainer = hydra.utils.instantiate(orig_cfg.trainer)

	# Instantiate the datamodule use a different config for data then for training
	if hasattr(orig_cfg, "data"):
		datamodule = hydra.utils.instantiate(cfg.data)
	else:
		datamodule = hydra.utils.instantiate(orig_cfg.datamodule)
  
	tra_dataloader = datamodule.train_dataloader()
	#val_dataloader = datamodule.train_dataloader()
	batch1 = next(iter(tra_dataloader))
	print("batch1:", batch1)
 
	v1, v2 = batch1
	# Direc pass
	w1 = T.cat([v1, v2], dim=1)
	w2 = v2
	if model.latent_norm:
		e1 = normalize(model.encoder1(w1))
		e2 = normalize(model.encoder2(w2))
	else:
		e1 = model.encoder1(w1)
		e2 = model.encoder2(w2)
	latent = T.cat([e1, e2], dim=1)
	recon = model.decoder(latent)

	# Reverse pass
	e1_p = e1[T.randperm(len(e1))]
	latent_p = T.cat([e1_p, e2], dim=1)
	recon_p = model.decoder(latent_p)
	w1_n = recon_p[:]

	w2_n = recon_p[:, v1.shape[1]:]
	if model.latent_norm:
		e1_n = normalize(model.encoder1(w1_n))
		e2_n = normalize(model.encoder2(w2_n))
	else:
		e1_n = model.encoder1(w1_n)
		e2_n = model.encoder2(w2_n)


	print(e1.shape, e2.shape)
	matrix = e1 @ e2.T
	plot_matrix(to_np(matrix), "e1 @ e2")
	plt.figure()
	plt.hist(to_np(matrix).flatten(), bins=100)
	plt.figure()
	plt.hist(np.diagonal(to_np(matrix)), bins=100)
	print(np.diagonal(to_np(matrix)))
