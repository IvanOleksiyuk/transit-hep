import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import logging
import hydra
from pathlib import Path
import os
from omegaconf import DictConfig, OmegaConf
from twinturbo.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config
log = logging.getLogger(__name__)

@hydra.main(
    version_base=None, config_path=str('../config'), config_name="twinturbo_reco_cons_DisCo_LHCO_CURTAINS"
)
def main(cfg: DictConfig) -> None:
    log.info("Setting up full job config")
    print_config(cfg)
    data_cfg = cfg.data.template_training.train_data
    dataset_name = "train_data_cathode"
    data_cfg.plotting_path = "plot/user/inspect_data/" + dataset_name
    log.info("Instantiating the data module")
    dataset = hydra.utils.instantiate(data_cfg)


if __name__ == "__main__":
    main()