import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import logging

import hydra
from omegaconf import DictConfig

from twinturbo.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config

log = logging.getLogger(__name__)

# TODO pyroot utils will remove the need for ../configs
@hydra.main(
    version_base=None, config_path=str(root / 'conf'), config_name="train"
)
def main(cfg: DictConfig) -> None:
    log.info("Setting up full job config")
    print_config(cfg)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.data)

if __name__ == "__main__":
    main()