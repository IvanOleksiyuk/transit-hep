import logging

import hydra
from matplotlib.backends.backend_pdf import PdfPages
from mltools.hydra_utils import print_hydra_config
from mltools.plotting import find_template_id
from mltools.utils import load_nested_array_dict_from_h5
from omegaconf import DictConfig

# setup logger
log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="plot", version_base=None)
def main(cfg: DictConfig) -> None:
    """Plot any kind of summarizing metric comparing either truth and any number of
    model predictions or summarizing them on their own."""
    print_hydra_config(cfg)

    log.info("Loading truth and predictions")
    truth = hydra.utils.call(cfg.load_predict_set)

    predictions = {}
    for source_id, prediction_path in cfg.io.prediction_paths.items():
        predictions[source_id] = load_nested_array_dict_from_h5(prediction_path)

    log.info("Create prediction summaries")
    summaries = {
        summary_id: hydra.utils.instantiate(
            summary_cfg, truth_ref=truth, pred_ref=predictions
        )
        for summary_id, summary_cfg in cfg.plot_def.prediction_summaries.items()
        # only instantiate summaries that are actually used in the layout
        if any(summary_id in row for row in cfg.plot_def.layout)
    }

    log.info("Instantiate templates")
    templates = hydra.utils.instantiate(cfg.plot_def.templates)

    with PdfPages(cfg.io.output_path) as fig_dest:
        log.info("Instantiating target and template")
        target = hydra.utils.instantiate(cfg.plot_def.target)

        log.info("Plot prediction summaries and save result")
        for summary_id, summary in summaries.items():
            log.info(f"Plot {summary_id}")
            template_id = find_template_id(
                summary_id, cfg.plot_def.summary_template_map
            )

            fig = target.setup_figure(cfg.plot_def.layout, summary_id)
            templates[template_id].plot(fig, summary)
            target.save_figure(fig, fig_dest)


if __name__ == "__main__":
    main()
