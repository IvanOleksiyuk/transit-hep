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
import twinturbo.scripts.evaluation as evaluation
import scripts.train_ptl as train_ptl
import twinturbo.scripts.generate_teplate as generate_teplate
import twinturbo.scripts.run_cwola_curtains2 as run_cwola_curtains2
log = logging.getLogger(__name__)

# TODO pyroot utils will remove the need for ../configs
@hydra.main(
    version_base=None, config_path=str('../config'), config_name="twinturbo_reco_cons0.001_adv_LHCO_CURTAINS1024b"
)
def main(cfg: DictConfig) -> None:
	## 1 - Create a separate folder for the experiment save all the relavant configs there
	# Create a folder for the experiment
	run_dir = Path(cfg.general.run_dir)
	os.makedirs(run_dir, exist_ok=True)
	os.makedirs(cfg.step_train_template.paths.full_path, exist_ok=True)
	print_config(cfg)
	OmegaConf.save(cfg, Path(cfg.general.run_dir, "full_config.yaml"), resolve=True)

	# run all the steps
	if cfg.do_train_template:
		log.info("Train a model for template generation")
		train_ptl.main(cfg.step_train_template)
	if cfg.do_export_template:
		log.info("Generate a template dataset using the model")
		name = cfg.step_export_template.output_name
		if hasattr(cfg.step_export_template, "signal_contamination_ns"):
			for cont_n in cfg.step_export_template.signal_contamination_ns:
				cfg.step_export_template["data"]["test_data"]["dataset2"]["processor_cfg"][2]["n_contamination"]=cont_n
				cfg.step_export_template["output_name"] = name + f"{cont_n}"
				generate_teplate.main(cfg.step_export_template)
		else:
			generate_teplate.main(cfg.step_export_template)

	if cfg.do_cwola:
		log.info("Train CWOLA model using the template dataset and the real data")
		run_cwola_curtains2.main(cfg.step_cwola)
	if cfg.do_evaluation:
		log.info("Evaluate the performance and plot the results")
		evaluation.main(cfg)

if __name__ == "__main__":
    main()
    

