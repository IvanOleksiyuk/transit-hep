import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")

import twinturbo
import logging
import wandb

import hydra
import pytorch_lightning as pl
import torch as T
from omegaconf import DictConfig

from twinturbo.src.utils.hydra_utils import instantiate_collection, log_hyperparameters, print_config, reload_original_config, save_config

log = logging.getLogger(__name__)

# TODO pyroot utils will remove the need for ../configs
@hydra.main(
    version_base=None, config_path=str('../conf'), config_name="train"
)
def main(cfg: DictConfig) -> None:
    wandb_key = open("/home/users/o/oleksiyu/WORK/keys/wandb.key", "r").read()
    wandb.login(key=wandb_key)
    log.info("Setting up full job config")
    if cfg.full_resume:
        cfg = reload_original_config(cfg)
    print_config(cfg)

    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    if cfg.precision:
        log.info(f"Setting matrix precision to: {cfg.precision}")
        T.set_float32_matmul_precision(cfg.precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    
    log.info("Instantiating the model")
    model = hydra.utils.instantiate(cfg.model, inpt_dim=datamodule.get_dim(), var_group_list=datamodule.get_var_group_list())
    log.info(model)

    if cfg.compile:
        log.info(f"Compiling the model using torch 2.0: {cfg.compile}")
        model = T.compile(model, mode=cfg.compile)

    log.info("Instantiating all callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating the loggers")
    loggers = instantiate_collection(cfg.loggers)

    log.info("Instantiating the trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        log.info("Logging all hyperparameters")
        log_hyperparameters(cfg, model, trainer)

    log.info("Saving config so job can be resumed")
    save_config(cfg)

    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
