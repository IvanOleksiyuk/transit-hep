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
import torch
from torch.nn.functional import normalize, mse_loss, cosine_similarity
from scipy.stats import pearsonr, spearmanr, kendalltau
import pickle
import dcor
from twinturbo.src.utils.hsic import HSIC_np, HSIC_torch
log = logging.getLogger(__name__)

def to_np(inpt: Union[torch.Tensor, tuple]) -> np.ndarray:
    """More consicse way of doing all the necc steps to convert a pytorch
    tensor to numpy array.

    - Includes gradient deletion, and device migration
    """
    if isinstance(inpt, (tuple, list)):
        return type(inpt)(to_np(x) for x in inpt)
    if inpt.dtype == torch.bfloat16:  # Numpy conversions don't support bfloat16s
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

	####uncomment
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
	scatter_alpha=1
	scatter_s=2
	if GlobalHydra().is_initialized():
		GlobalHydra().clear()
	hydra.initialize(version_base=None, config_path= "../config")

	log.info("Loading run information")
	cfg = cfg.step_export_template
	print(cfg.paths.full_path)
	orig_cfg = reload_original_config(cfg, get_best=cfg.get_best)
	cfg = orig_cfg

	plot_path= cfg["paths"]["output_dir"]+"/../plots/"

	log.info("Loading best checkpoint")
	#device = "cuda" if torch.cuda.is_available() else "cpu"
	device = "cpu"
	model_class = hydra.utils.get_class(orig_cfg.model._target_)
	model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location=device)
	#model.to(device)
 
	#log.info("Instantiating original trainer")
	#trainer = hydra.utils.instantiate(orig_cfg.trainer)

	# Instantiate the datamodule use a different config for data then for training
	datamodule = hydra.utils.instantiate(cfg.data.datamodule)
  
	tra_dataloader = datamodule.train_dataloader()
	batch1 = next(iter(tra_dataloader))
	#print("batch1:", batch1)

	batch_size=batch1[0].shape[0]
	w1, w2 = batch1
	m_dn = w2
	if model.use_m:
		x = w1[:, :-w2.shape[1]]
	else:
		x = w1

	e1, e2 = model.encode(w1, w2)
	latent = torch.cat([e1, e2], dim=1)
	recon = model.decoder(latent)
	
	# Reverse pass
	e1_p = e1[torch.randperm(batch_size)]
	latent_p = torch.cat([e1_p, e2], dim=1)
	recon_p = model.decoder(latent_p)
	x_n = recon_p[:, :x.shape[1]]
	m_n = recon_p[:, x.shape[1]:]
	
	if model.use_m:
		w1_n = torch.cat([x_n, m_n*model.use_m], dim=1)
	else:
		w1_n = x_n
	w2_n = m_n

	e1_n, e2_n = model.encode(w1_n, w2_n)

	matrix = e1 @ e2.T
 
	plt.figure()
	plot_matrix(to_np(matrix), "e1 @ e2")
	plt.savefig(plot_path+"e1_at_e2_matrix.png")
	plt.figure()
	plt.hist(to_np(matrix).flatten(), bins=100)
	plt.title("One batch latent embedding product <e1, e2>")
	plt.xlabel("<e1, e2>")
	plt.savefig(plot_path+"e1_at_e2_hist.png")
	plt.figure()
	plt.title("One batch latent embedding product <e1, e2> only diagonal")
	plt.xlabel("<e1, e2>")
	plt.hist(np.diagonal(to_np(matrix)), bins=100)
	plt.savefig(plot_path+"e1_at_e2_diag_hist.png")
 
	bins= np.linspace(-3, 3, 30)
	for i in range(w1.shape[1]):
		plt.figure()
		plt.hist(to_np(w1[:, i]), bins=bins, histtype='step', label="input")
		plt.hist(to_np(recon[:, i]), bins=bins, histtype='step', label="reconstruction")
		plt.xlabel(f"dim{i}")
		plt.legend()
		plt.savefig(plot_path+f"w1_reco_hist_{i}.png")
		plt.figure()
		plt.scatter(to_np(w1[:, i]), to_np(recon[:, i]), alpha=scatter_alpha, s=scatter_s)
		plt.xlabel(f"dim{i}_input")
		plt.xlabel(f"dim{i}_reco")
		plt.legend()
		plt.savefig(plot_path+f"w1_reco_scater_{i}.png")
	# Plot linear correlateion plots for the latent space
	one_corretation_plot=True
	os.makedirs(plot_path+"corerlations/", exist_ok=True)
	person_correlations, spearman_correlations, kendalltaus = plot_correlation_plots(e1, e2, plot_path, one_corretation_plot=True, name="latent_space_correlations", c=m_dn)
    
	# Mass correlation plots
	person_correlations_mass = []
	spearman_correlations_mass = []
	kendalltaus_mass = []
	if one_corretation_plot:
		fig, axes = plt.subplots(1, e1.shape[1],  figsize=(3*e1.shape[1], 3))
		fig.suptitle('Scatter plots with Pearson Correlation', fontsize=16)
		for i in range(e1.shape[1]):
			axes[i].scatter(to_np(e1[:, i]), to_np(m_dn), marker="o", label="e1", c=m_dn, cmap="viridis")
			pearson_correlation, p_value = pearsonr(to_np(e1[:, i]), to_np(m_dn).T[0])
			person_correlations_mass.append(pearson_correlation)
			spearman_correlation, p_value = spearmanr(to_np(e1[:, i]), to_np(m_dn).T[0])
			spearman_correlations_mass.append(spearman_correlation)
			kendalltau_correlation, p_value = kendalltau(to_np(e1[:, i]), to_np(m_dn).T[0])
			kendalltaus_mass.append(kendalltau_correlation)
			axes[i].set_title(f"Corr={pearson_correlation:.3f}")
			axes[i].set_xlabel(f"dim{i} e1")
			axes[i].set_ylabel(f"mjj")
		plt.tight_layout()
		plt.savefig(plot_path+"corerlations/"+"latent_space_e1_mass_correlations.png")

	# Same mass
	plt.figure()	
	plt.scatter(np.diagonal(to_np(matrix)), to_np(m_dn), alpha=scatter_alpha, s=scatter_s)
	plt.xlabel("e1 @ e2 diagonal elements")
	plt.ylabel("mjj")
	plt.savefig(plot_path+"e1_at_e2_diag_vs_mjj.png")
 
	# Different mass
	n = to_np(matrix).shape[0]
	plt.figure()	
	diag_mask = np.eye(n, dtype=bool)

	# Invert the mask to get the non-diagonal elements
	non_diag_mask = np.logical_not(diag_mask)
	non_diag_elements = to_np(matrix)[non_diag_mask]
	m_1_non_diag = np.tile(m_dn, (1, n))[non_diag_mask]
	m_2_non_diag = np.tile(m_dn.T, (n, 1))[non_diag_mask]
	plt.figure()
	plt.scatter(non_diag_elements, m_1_non_diag, alpha=scatter_alpha, s=scatter_s)
	plt.xlabel("e1 @ e2 non-diagonal elements")
	plt.ylabel("mjj1")
	plt.savefig(plot_path+"e1_at_e2_non_diag_vs_mjj1.png")
	plt.figure()
	plt.scatter(non_diag_elements, m_2_non_diag, alpha=scatter_alpha, s=scatter_s)
	plt.xlabel("e1 @ e2 non-diagonal elements")
	plt.ylabel("mjj2")
	plt.savefig(plot_path+"e1_at_e2_non_diag_vs_mjj2.png")
	plt.figure()
	plt.scatter(non_diag_elements, m_1_non_diag-m_2_non_diag, alpha=scatter_alpha, s=scatter_s)
	plt.xlabel("e1 @ e2 non-diagonal elements")
	plt.ylabel("mjj1 - mjj2")
	plt.savefig(plot_path+"e1_at_e2_non_diag_vs_mjj1-mjj2.png")

	# Compute some numerical metrics as a summary about model performance
	results = {"max_abs_pearson": np.max(np.abs(person_correlations)), "min_abs_pearson": np.min(np.abs(person_correlations)), "mean_abs_pearson": np.mean(np.abs(person_correlations))}
	results.update({"max_abs_spearman": np.max(np.abs(spearman_correlations)), "min_abs_spearman": np.min(np.abs(spearman_correlations)), "mean_abs_spearman": np.mean(np.abs(spearman_correlations))})
	results["kernel_pearson"] = None
	results["hilbert_schmidt"] = HSIC_torch(e1, e2, cuda=False).detach().cpu().numpy()
	results["DisCo"] = dcor.distance_correlation(to_np(e1), to_np(e2))
	
	pickle.dump(results, open(plot_path+"results.pkl", "wb"))
	with open(plot_path+"results.txt", "w") as f:
		for key, value in results.items():
			f.write(f"{key}: {value}\n")
	for key, value in results.items():
		print(key, value)
	


def plot_correlation_plots(e1, e2, plot_path, name, c=None, one_corretation_plot=True):
	person_correlations =np.zeros((e1.shape[1], e2.shape[1]))
	spearman_correlations = np.zeros((e1.shape[1], e2.shape[1]))
	kendalltaus = np.zeros((e1.shape[1], e2.shape[1]))
	if one_corretation_plot:
		fig, axes = plt.subplots(e1.shape[1], e2.shape[1], figsize=(3*e1.shape[1], 3*e1.shape[1]))
		fig.suptitle('Scatter plots with Pearson Correlation', fontsize=16)
		for i in range(e1.shape[1]):
			for j in range(e1.shape[1]):
				axes[i, j].scatter(to_np(e1[:, i]), to_np(e2[:, j]), marker="o", label="e1", c=c, cmap="viridis")
				pearson_correlation, p_value = pearsonr(to_np(e1[:, i]), to_np(e2[:, j]))
				person_correlations[i, j] = pearson_correlation
				spearman_correlation, p_value = spearmanr(to_np(e1[:, i]), to_np(e2[:, j]))
				spearman_correlations[i, j] = spearman_correlation	
				kendalltau_correlation, p_value = kendalltau(to_np(e1[:, i]), to_np(e2[:, j]))
				kendalltaus[i, j] = kendalltau_correlation
				axes[i, j].set_title(f"Corr={pearson_correlation:.3f}")
				axes[i, j].set_xlabel(f"dim{i} e1")
				axes[i, j].set_ylabel(f"dim{j} e2")
		plt.tight_layout()
		plt.savefig(plot_path+"corerlations/"+name+".png")
	else:
		for dim1 in range(e1.shape[1]):
			for dim2 in range(e2.shape[1]):
				plt.figure()
				plt.scatter(to_np(e1[:, dim1]), to_np(e2[:, dim2]), marker="o", label="e1", c=c, cmap="viridis")
				pearson_correlation, p_value = pearsonr(to_np(e1[:, i]), to_np(e2[:, j]))
				person_correlations[i, j] = pearson_correlation
				spearman_correlation, p_value = spearmanr(to_np(e1[:, i]), to_np(e2[:, j]))
				spearman_correlations[i, j] = spearman_correlation	
				kendalltau_correlation, p_value = kendalltau(to_np(e1[:, i]), to_np(e2[:, j]))
				kendalltaus[i, j] = kendalltau_correlation
				plt.title(f"Latent space (color=m) pearson={pearson_correlation}")
				plt.xlabel(f"dim{dim1} e1")
				plt.ylabel(f"dim{dim2} e2")
				plt.savefig(plot_path+"corerlations/"+f"{name}_{dim1}_{dim2}.png")
    
	return person_correlations, spearman_correlations, kendalltaus