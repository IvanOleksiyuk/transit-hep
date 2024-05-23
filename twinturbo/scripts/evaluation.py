import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

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
from scipy.stats import pearsonr
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

@hydra.main(
    version_base=None, config_path=str('../conf'), config_name="evaluate"
)
def main(cfg):
	print("Starting evaluation")
	# Get two dataframes to compare	
	original_data = hydra.utils.instantiate(cfg.step_evaluate.original_data).data["data"]
	target_data = hydra.utils.instantiate(cfg.step_evaluate.target_data).data["data"]
	template_data = pd.read_hdf(cfg.step_export_template.paths.full_path + "/outputs/template_sample.h5")

	variables = original_data.columns.tolist()
	print(len(target_data.to_numpy()))
	print(len(template_data.to_numpy()))

	pltt.plot_feature_spread(
    target_data[variables].to_numpy(),
    template_data[variables].to_numpy(),
    original_data = original_data[variables].to_numpy(),
    feature_nms = variables,
    save_dir=Path(cfg.general.run_dir),
    plot_mode="")
 	
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
	hydra.initialize(version_base=None, config_path= "../config")
	cfg = hydra.compose(config_name="twinturbo.yaml")
	log.info("Loading run information")
	cfg = cfg.export_template
	print(cfg.paths.full_path)
	orig_cfg = reload_original_config(cfg, get_best=True)
	cfg = orig_cfg

	plot_path= cfg["paths"]["output_dir"]+"/../plots/"

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

	matrix = e1 @ e2.T
 
	plt.figure()
	plot_matrix(to_np(matrix), "e1 @ e2")
	plt.savefig(plot_path+"e1_at_e2_matrix.png")
	plt.figure()
	plt.hist(to_np(matrix).flatten(), bins=100)
	plt.savefig(plot_path+"e1_at_e2_hist.png")
	plt.figure()
	plt.hist(np.diagonal(to_np(matrix)), bins=100)
	plt.savefig(plot_path+"e1_at_e2_diag_hist.png")
 
	bins= np.linspace(-3, 3, 30)
	for i in range(recon.shape[1]):
		plt.figure()
		plt.hist(to_np(T.cat([v1, v2], dim=1)[:, i]), bins=bins, histtype='step')
		plt.hist(to_np(recon[:, i]), bins=bins, histtype='step')
		plt.savefig(plot_path+f"e1_at_e2_{i}.png")
  
	# Plot linear correlateion plots for the latent space
	one_corretation_plot=True
	if one_corretation_plot:
		fig, axes = plt.subplots(e1.shape[1], e1.shape[1], figsize=(3*e1.shape[1], 3*e1.shape[1]))
		fig.suptitle('Scatter plots with Pearson Correlation', fontsize=16)
		for i in range(e1.shape[1]):
			for j in range(e1.shape[1]):
				axes[i, j].scatter(to_np(e1[:, i]), to_np(e2[:, j]), marker="o", label="e1", c=v2, cmap="viridis")
				pearson_correlation, p_value = pearsonr(to_np(e1[:, i]), to_np(e2[:, j]))
				axes[i, j].set_title(f"Corr={pearson_correlation:.3f}")
				axes[i, j].set_xlabel(f"dim{i} e1")
				axes[i, j].set_ylabel(f"dim{j} e2")
		plt.tight_layout()
		plt.savefig(plot_path+"corerlations/"+"latent_space_correlations.png")
	else:
		for dim1 in range(e1.shape[1]):
			for dim2 in range(e1.shape[1]):
				plt.figure()
				plt.scatter(to_np(e1[:, dim1]), to_np(e2[:, dim2]), marker="o", label="e1", c=v2, cmap="viridis")
				pearson_correlation, p_value = pearsonr(to_np(e1[:, dim1]), to_np(e2[:, dim2]))
				plt.title(f"Latent space (color=m) pearson={pearson_correlation}")
				plt.xlabel(f"dim{dim1} e1")
				plt.ylabel(f"dim{dim2} e2")
				os.makedirs(plot_path+"corerlations/", exist_ok=True)
				plt.savefig(plot_path+"corerlations/"+f"latent_space_{dim1}_{dim2}.png")
    
	# Mass correlation plots
	if one_corretation_plot:
		fig, axes = plt.subplots(1, e1.shape[1],  figsize=(3*e1.shape[1], 3))
		fig.suptitle('Scatter plots with Pearson Correlation', fontsize=16)
		for i in range(e1.shape[1]):
			axes[i].scatter(to_np(e1[:, i]), to_np(v2), marker="o", label="e1", c=v2, cmap="viridis")
			pearson_correlation, p_value = pearsonr(to_np(e1[:, i]), to_np(v2).T[0])
			axes[i].set_title(f"Corr={pearson_correlation:.3f}")
			axes[i].set_xlabel(f"dim{i} e1")
			axes[i].set_ylabel(f"mjj")
		plt.tight_layout()
		plt.savefig(plot_path+"corerlations/"+"latent_space_e1_mass_correlations.png")

	# Same mass
	plt.figure()	
	plt.scatter(np.diagonal(to_np(matrix)), to_np(v2))
	plt.savefig(plot_path+"e1_at_e2_diag_vs_mjj.png")
 
	# Different mass
	n = to_np(matrix).shape[0]
	plt.figure()	
	diag_mask = np.eye(n, dtype=bool)

	# Invert the mask to get the non-diagonal elements
	non_diag_mask = ~diag_mask
	non_diag_elements = to_np(matrix)[non_diag_mask]
	m_1_non_diag = np.tile(v2, (n, 1))[non_diag_mask]
	m_2_non_diag = np.tile(v2, (n, 1)).T[non_diag_mask]
	plt.figure()
	plt.scatter(non_diag_elements, m_1_non_diag)
	plt.savefig(plot_path+"e1_at_e2_non_diag_vs_mjj1.png")
	plt.figure()
	plt.scatter(non_diag_elements, m_2_non_diag)
	plt.savefig(plot_path+"e1_at_e2_non_diag_vs_mjj2.png")
	plt.figure()
	plt.scatter(non_diag_elements, m_1_non_diag-m_2_non_diag)
	plt.savefig(plot_path+"e1_at_e2_non_diag_vs_mjj1-mjj2.png")
 
	


