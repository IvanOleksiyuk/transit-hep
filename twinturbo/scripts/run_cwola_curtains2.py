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
    
    datasets = hydra.utils.instantiate(cfg.datasets)
    datasr = datasets.datasr
    print("datasr len:", len(datasr))
    datasr.write_npy_single(file_path_str=cfg.cwola_path+cfg.cwola_subfolders+"window_3100_3300__3700_3900/dope_3000/sr.npy", key="data")
    datasr.write_features_txt(file_path_str=cfg.cwola_path+"window_3100_3300__3700_3900/dope_3000/features.txt", key="data")
    #datasr.write_npy(file_dir=cfg.cwola_path+"window_3100_3300__3700_3900/dope_3000/", keys=["data"], save_file_names=["sr"])
    
    template = datasets.template
    print("template len:", len(template))
    template.write_npy_single(file_path_str=cfg.cwola_path+cfg.cwola_subfolders+"window_3100_3300__3700_3900/dope_3000/template.npy", key="template")
    #template.write_npy(file_dir=cfg.cwola_path+"window_3100_3300__3700_3900/dope_3000/", keys=["template"], save_file_names=["template"])

    extra_signal = datasets.extra_signal
    print("extra_signal len:", len(extra_signal))
    extra_signal.write_npy_single(file_path_str=cfg.cwola_path+cfg.cwola_subfolders+"window_3100_3300__3700_3900/dope_3000/extra_signal.npy", key="data")
    #extra_signal.write_npy(file_dir=cfg.cwola_path+"window_3100_3300__3700_3900/dope_3000/", keys=["data"], save_file_names=["extra_signal"])
    

    #python libs_snap/linearanomaly/cwola.py --input_path=twinturbo/workspaces/dev/twinTURBO_DisCo_LHCO_CWOLA/cwola/ --mode=standard --num_signal=3000 --sideband_1=3100_3300 --sideband_2=3700_3900 --num_folds=5 --max_iter=250 --early_stopping=True --validation_fraction=0.1 --class_weight=balanced --num_ensemble=5 --seed=0
#python libs_snap/linearanomaly/cwola.py --input_path=/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/workspaces/adv1_gauss_corr_4_gap_twinturbo_usem_addgapmass/twinturbo_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS1024b/cwola/ --mode=standard --num_signal=3000 --sideband_1=3100_3300 --sideband_2=3700_3900 --num_folds=5 --max_iter=250 --early_stopping=1 --validation_fraction=0.1 --class_weight=balanced --num_ensemble=5 --seed=0


    command = [
        "python", 
        "libs/curtains2/cwola.py",
        "--input_path=" + cfg.cwola_path,
        "--mode=" + cfg.mode,
        "--num_signal=" + str(cfg.num_signal),
        "--sideband_1=" + str(cfg.sideband_1),
        "--sideband_2=" + str(cfg.sideband_2),
        "--num_folds=" + str(cfg.num_folds),
        "--max_iter=" + str(cfg.max_iter),
        #"--early_stopping=" + str(cfg.early_stopping),
        "--validation_fraction=" + str(cfg.validation_fraction),
        "--class_weight=" + str(cfg.class_weight),
        "--num_ensemble=" + str(cfg.num_ensemble),
        "--seed=" + str(cfg.seed),
        ]
    print(" ".join(command))
    # run cwola
    return_code = subprocess.run(command) 
    print(return_code)   
    log.info("<<<FINISH CWOLA SCRIPT>>>")
    
    # analyse results of cwola
    
if __name__ == "__main__":
    main()