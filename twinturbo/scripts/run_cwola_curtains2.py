import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
import os
import logging

import hydra
from omegaconf import DictConfig

import subprocess

log = logging.getLogger(__name__)

# TODO pyroot utils will remove the need for ../configs
@hydra.main(
    version_base=None, config_path=str('../config/step_cwola'), config_name="cwola"
)
def main(cfg: DictConfig) -> None:
    # prepare data for running cwola 
    log.info("<<<START CWOLA SCRIPT>>>")
    os.makedirs(cfg.cwola_path, exist_ok=True)
    
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    dataset = datamodule.train_data
    dataset.write_npy(cfg.cwola_path)
    
    # run cwola
    return_code = subprocess.run([
        "python", 
        "curtains2/cwola.py",
        "--input_path=" + cfg.cwola_path,
        # "--mode=" + cfg.mode,
        # "--num_signal=" + str(cfg.num_signal),
        # "--sideband_1=" + str(cfg.sideband_1),
        # "--sideband_2=" + str(cfg.sideband_2),
        # "--num_folds=" + str(cfg.num_folds),
        # "--max_iter=" + str(cfg.max_iter),
        # "--early_stopping=" + str(cfg.early_stopping),
        # "--validation_fraction=" + str(cfg.validation_fraction),
        # "--class_weight=" + str(cfg.class_weight),
        # "--num_ensemble=" + str(cfg.num_ensemble),
        # "--seed=" + str(cfg.seed),
        ]) 
    print(return_code)   
    log.info("<<<FINISH CWOLA SCRIPT>>>")
    
    # analyse results of cwola
    
if __name__ == "__main__":
    main()