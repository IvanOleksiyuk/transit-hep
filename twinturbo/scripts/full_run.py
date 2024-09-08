# This is a script to do one full LHCO analysis. 
# It is a combination of the several steps.

# 1 - Create a separate folder for the experiment save all the relavant configs there 
# 2 - train/create a model that will provide a us with a template (e.g. CATHODE, CURTAINS)
# 3 - generate a template dataset using the model
# 4 - evaluate the performance of the template generation model
# 4 - train cwola
# 5 - evaluate the performance and plot the results
# 6 - produce a set of final plots and tables for one run

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
import twinturbo.scripts.cwola_evaluation as cwola_evaluation
import twinturbo.scripts.plot_compare as plot_compare
import twinturbo.scripts.export_latent_space as export_latent_space
from datetime import datetime
log = logging.getLogger(__name__)

# TODO pyroot utils will remove the need for ../configs
@hydra.main(
    version_base=None, config_path=str('../config'), config_name="TTL_reco_LLCR_1024b_lr2-4_dl"
)

def main(cfg: DictConfig) -> None:
    log.info("<<<START FULL RUN>>>")
    ## 1 - Create a separate folder for the experiment save all the relavant configs there
    # Create a folder for the experiment
    run_dir = Path(cfg.general.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(cfg.step_train_template.paths.full_path, exist_ok=True)
    #print_config(cfg)
    OmegaConf.save(cfg, Path(cfg.general.run_dir, "full_config.yaml"), resolve=True)

    # run all the steps
    if cfg.do_train_template:
        start_time = datetime.now()
        log.info("===================================")
        log.info(f"Start: Train a model that will provide a us with a template")
        train_ptl.main(cfg.step_train_template)
        log.info(f"Finish: Train a model that will provide a us with a template. Time taken: {datetime.now() - start_time}")
        log.info(f"===================================")
    
    if cfg.do_export_template:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Generate a template dataset using the model")
        name = cfg.step_export_template.output_name
        if hasattr(cfg.step_export_template, "signal_contamination_ns"):
            for cont_n in cfg.step_export_template.signal_contamination_ns:
                for processor in cfg.step_export_template["data"]["test_data"]["dataset2"]["processor_cfg"]:
                    if processor["_target_"] == "twinturbo.src.data.processors.SignalContamination":
                        processor["n_sig"] = cont_n
                cfg.step_export_template["output_name"] = name + f"{cont_n}"
                generate_teplate.main(cfg.step_export_template)
        else:
            generate_teplate.main(cfg.step_export_template)
        log.info(f"Finish: Generate a template dataset using the model. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
    
    if cfg.do_export_latent:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start:Generate latent representation of events in SR and Sidebands")
        name = cfg.step_export_latent.export_latent_all.output_name
        export_latent_space.main(cfg.step_export_latent.export_latent_all)
        log.info(f"Finish: Generate latent representation of events in SR and Sidebands Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        
    if cfg.do_evaluation:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Evaluate the performance and plot the results")
        evaluation.main(cfg)
        log.info(f"Finish: Evaluate the performance and plot the results. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        
    if cfg.do_cwola:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Train CWOLA model using the template dataset and the real data")
        if hasattr(cfg.step_cwola, "several_confs"):
            for conf in cfg.step_cwola.several_confs:
                run_cwola_curtains2.main(conf)
        else:
            run_cwola_curtains2.main(cfg.step_cwola)
        log.info(f"Finish: Train CWOLA model using the template dataset and the real data. Time taken: {datetime.now() - start_time}")
        log.info("===================================")

    if cfg.do_evaluate_cwola:
        log.info("Evaluate the performance of the CWOLA model")
        log.info("Train CWOLA model using the template dataset and the real data")
        if hasattr(cfg.step_cwola, "several_confs"):
            for conf in cfg.step_cwola.several_confs:
                cwola_evaluation.main(conf)
        else:
            cwola_evaluation.main(cfg.step_cwola)

    if cfg.do_plot_compare:
        log.info("Produce a set of final plots and tables for one run")
        plot_compare.main(cfg.step_plot_compare)
    
    log.info("<<<END FULL RUN>>>")


if __name__ == "__main__":
    main()
    

