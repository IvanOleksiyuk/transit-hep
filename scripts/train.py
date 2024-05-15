import logging

import hydra
import lightning as L
import torch as T
from mltools.hydra_utils import link_best_model, log_hyperparameters, print_hydra_config
from omegaconf import DictConfig

# setup logger
log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train and save a model.

    Automatically resumes if it finds a checkpoint
    """
    print_hydra_config(cfg)

    if cfg.seed:
        log.info(f"Setting seed to: {cfg.seed}")
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Setting matrix precision to: {cfg.common.matmul_precision}")
    T.set_float32_matmul_precision(cfg.common.matmul_precision)

    log.info("Instantiating the data module")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # use a mock sample of the data to figure out model in/out dims
    log.info("Instantiating the model")
    model = hydra.utils.instantiate(cfg.model, mock_sample=datamodule.mock_sample())
    if cfg.common.model_compile_mode:
        log.info(
            f"Compiling the model using torch 2.0: {cfg.common.model_compile_mode}"
        )
        model = T.compile(model, mode=cfg.common.model_compile_mode)

    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    if cfg.trainer.logger:
        log.info("Logging all hyperparameters")
        log_hyperparameters(cfg, model, trainer)
        log.info(model)

    log.info("Train")
    trainer.fit(model, datamodule, ckpt_path=cfg.init_ckpt_path)

    # We want to have a nice and predictable model save path for predicting, no
    # matter what we do regarding checkpointing; this is also practically relevant,
    # so snakemake doesn't think that the training ended because it finds a
    # checkpoint from some intermediate epoch
    if trainer.state.status == "finished":
        # only create this link if the training finished successfully,
        # otherwise we don't have a result, so don't save one
        log.info("Create link to best model checkpoint")
        link_best_model(trainer.checkpoint_callback, cfg.io.result_path)


if __name__ == "__main__":
    main()
