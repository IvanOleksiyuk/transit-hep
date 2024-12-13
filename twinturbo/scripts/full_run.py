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
import twinturbo.scripts.train_ptl as train_ptl
import twinturbo.scripts.generate_teplate as generate_teplate
import scripts.run_cwola as run_cwola
import twinturbo.scripts.cwola_evaluation as cwola_evaluation
import twinturbo.scripts.plot_compare as plot_compare
import twinturbo.scripts.export_latent_space as export_latent_space
import twinturbo.scripts.time_chart as time_chart
from datetime import datetime
import subprocess
log = logging.getLogger(__name__)

# TODO pyroot utils will remove the need for ../configs
def get_git_hash():
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).strip().decode("utf-8") #"cd", "twinturbo", "&&",
        print(f"Git hash: {git_hash}")
        return git_hash
    except subprocess.CalledProcessError as e:
        print("Error retrieving git hash:", e.output.decode("utf-8"))
        return None

@hydra.main(
    version_base=None, config_path=str('../config'), config_name="TRANSITv0v1_LHCO_test"
)
def main(cfg: DictConfig) -> None:
    log.info("<<<START FULL RUN>>>")
    ## 1 - Create a separate folder for the experiment save all the relavant configs there
    # Create a folder for the experiment
    run_dir = Path(cfg.general.run_dir)
    
    # create a summary folder and a file to save the runtime of each step
    summary_dir = run_dir / "summary"
    os.makedirs(summary_dir, exist_ok=True)
    rutime_file = summary_dir / "runtime.txt"
    
    with open(rutime_file, 'w') as file:
        file.write('Start time: {}\n'.format(datetime.now()))
    
    # Save git hash to a file
    git_hash_file = summary_dir / "git_hash.txt"
    git_hash=get_git_hash()
    if git_hash:
        with open(git_hash_file, 'w') as file:
            file.write(git_hash)
    
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
        with open(rutime_file, 'a') as file:
            file.write('Train template: {}\n'.format(datetime.now() - start_time))
    
    if cfg.do_export_template:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Generate a template dataset using the model")
        name = cfg.step_export_template.output_name
        generate_teplate.main(cfg.step_export_template)
        log.info(f"Finish: Generate a template dataset using the model. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        with open(rutime_file, 'a') as file:
            file.write('Generate template: {}\n'.format(datetime.now() - start_time))
    
    if hasattr(cfg, "do_transport_sideband") and cfg.do_transport_sideband:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Generate a template dataset using the model")
        name = cfg.step_export_template.output_name
        generate_teplate.main(cfg.step_export_SB1)
        generate_teplate.main(cfg.step_export_SB2)
        log.info(f"Finish: Generate a template dataset using the model. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        with open(rutime_file, 'a') as file:
            file.write('Generate template: {}\n'.format(datetime.now() - start_time))
    
    if hasattr(cfg, "do_export_latent") and cfg.do_export_latent:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start:Generate latent representation of events in SR and Sidebands")
        name = cfg.step_export_latent.export_latent_all.output_name
        export_latent_space.main(cfg.step_export_latent.export_latent_all)
        log.info(f"Finish: Generate latent representation of events in SR and Sidebands Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        with open(rutime_file, 'a') as file:
            file.write('Generate latent: {}\n'.format(datetime.now() - start_time))
        
    if cfg.do_evaluation:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Evaluate the performance and plot the results")
        evaluation.main(cfg)
        log.info(f"Finish: Evaluate the performance and plot the results. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        with open(rutime_file, 'a') as file:
            file.write('Evaluate: {}\n'.format(datetime.now() - start_time))
        
    if cfg.do_cwola:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Train CWOLA model using the template dataset and the real data")
        if hasattr(cfg.step_cwola, "several_confs"):
            for conf in cfg.step_cwola.several_confs.values():
                run_cwola.main(conf)
        else:
            run_cwola.main(cfg.step_cwola)
        log.info(f"Finish: Train CWOLA model using the template dataset and the real data. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        with open(rutime_file, 'a') as file:
            file.write('Train CWOLA: {}\n'.format(datetime.now() - start_time))

    if cfg.do_evaluate_cwola:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Start: Evaluate the performance of the CWOLA model")
        if hasattr(cfg.step_cwola, "several_confs"):
            for conf in cfg.step_cwola.several_confs:
                cwola_evaluation.main(conf)
        else:
            cwola_evaluation.main(cfg.step_cwola)
        log.info(f"Finish: Evaluate the performance of the CWOLA model. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        with open(rutime_file, 'a') as file:
            file.write('Evaluate CWOLA: {}\n'.format(datetime.now() - start_time))

    if cfg.do_plot_compare:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Produce a set of final plots and tables for one run")
        plot_compare.main(cfg.step_plot_compare)
        log.info(f"Finish: Produce a set of final plots and tables for one run. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        with open(rutime_file, 'a') as file:
            file.write('Plot compare: {}\n'.format(datetime.now() - start_time))
        
    if hasattr(cfg, "do_cleanup") and cfg.do_cleanup:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Clean up the workspace")
        
    
    if hasattr(cfg, "do_summary_plots") and cfg.do_summary_plots:
        start_time = datetime.now()
        log.info("===================================")
        log.info("Produce a set of summary plots")
        time_chart.main(rutime_file, save_path=summary_dir / "time_plot.png")
        log.info(f"Finish: Produce a set of summary plots. Time taken: {datetime.now() - start_time}")
        log.info("===================================")
        with open(rutime_file, 'a') as file:
            file.write('Summary plots: {}\n'.format(datetime.now() - start_time))
        
            
    
    log.info("<<<END FULL RUN>>>")


if __name__ == "__main__":
    main()
    

