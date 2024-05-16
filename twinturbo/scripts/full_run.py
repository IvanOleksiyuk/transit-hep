# This is a script to do one full LHCO analysis. 
# It is a combination of the several steps.

# 1 - Create a separate folder for the experiment save all the relavant configs there 
# 2 - train/create a model that will provide a us with a template (e.g. CATHODE, CURTAINS)
# 3 - generate a template dataset using the model
# 4 - train cwola
# 5 - evaluate the performance and plot the results 
import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import logging
import hydra
from pathlib import Path
import os
from omegaconf import DictConfig, OmegaConf
from twinturbo.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config
import train
import generate_teplate
import export
import pandas as pd
import twinturbo.src.utils.plotting as pltt
import numpy as np
from twinturbo.src.data.lhco_simple import LHCOInMemoryDataset
log = logging.getLogger(__name__)

# TODO pyroot utils will remove the need for ../configs
@hydra.main(
    version_base=None, config_path=str('../config'), config_name="twinturbo"
)
def main(cfg: DictConfig) -> None:
	## 1 - Create a separate folder for the experiment save all the relavant configs there
	# Create a folder for the experiment
	run_dir = Path(cfg.general.run_dir)
	os.makedirs(run_dir, exist_ok=True)
	os.makedirs(cfg.train_template.paths.full_path, exist_ok=True)
	print_config(cfg)
	OmegaConf.save(cfg, Path(cfg.general.run_dir, "full_run_config.yaml"), resolve=True)

	# run all the steps
	if cfg.do_train_template:
		log.info("Train a model for template generation")
		train.main(cfg.train_template)
	if cfg.do_export_template:
		log.info("Generate a template dataset using the model")
		generate_teplate.main(cfg.export_template)
	if cfg.do_train_cwola:
		log.info("Train CWOLA model using the template dataset and the real data")
		train.main(cfg.train_cwola)
	if cfg.do_evaluate_cwola:
		log.info("Evaluate the performance and plot the results")
		export.main(cfg.evaluate_cwola)
	if cfg.do_evaluation:
		log.info("Evaluate the performance and plot the results")
		evaluation(cfg)

def evaluation(cfg):
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
    # feature_nms=None,
    # tag=None,
    # 
    # plot_mode="diagnose",
    # combined=False,
    # x_bounds=None,
    # shuffle=True)
	#plt.savefig(cfg.general.run_dir + "/feature_spread.png")
	#print(len(template_data[0]))

if __name__ == "__main__":
    main()
    

