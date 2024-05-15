import logging
import tempfile

import hydra
import torch as T
from mltools.hydra_utils import print_hydra_config
from mltools.utils import save_nested_array_dict_as_h5
from omegaconf import DictConfig

# setup logger
log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="predict", version_base=None)
def main(cfg: DictConfig) -> None:
    """Load a model and the associated datamodule from a checkpoint and generate a
    prediction of the trained model."""
    print_hydra_config(cfg)

    log.info(f"Setting matrix precision to: {cfg.common.matmul_precision}")
    T.set_float32_matmul_precision(cfg.common.matmul_precision)

    log.info("Loading datamodule from checkpoint")
    datamodule = hydra.utils.call(cfg.load_datamodule)

    log.info("Loading model from checkpoint")
    model = hydra.utils.call(cfg.load_model, mock_sample=datamodule.mock_sample())
    if cfg.common.model_compile_mode:
        log.info(
            f"Compiling the model using torch 2.0: {cfg.common.model_compile_mode}"
        )
        model = T.compile(model, mode=cfg.common.model_compile_mode)

    log.info("Instantiating the trainer")
    # we are just predicting, so we don't care about logs
    with tempfile.TemporaryDirectory() as tempdir:
        predictor = hydra.utils.instantiate(cfg.predictor, default_root_dir=tempdir)

        log.info("Predicting")
        batches = predictor.predict(model, datamodule)

    log.info("Inverting datamodule setup on prediction")
    pred = datamodule.invert_setup_on_prediction(batches)

    log.info("Saving prediction")
    save_nested_array_dict_as_h5(cfg.io.predictions_save_path, pred)


if __name__ == "__main__":
    main()
