# This script  just build SIC curves and simmilar plots.

import pyrootutils
root = pyrootutils.setup_root(search_from=__file__, pythonpath=True, cwd=True, indicator=".project-root")
import logging
import hydra
from omegaconf import DictConfig
import os
from pathlib import Path 
import pandas as pd
log = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# TODO pyroot utils will remove the need for ../configs
@hydra.main(
    version_base=None, config_path=str('../config'), config_name="twinturbo_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS1024b"
)
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.run_dir)
    for seed in cfg.seeds:
        plot_path = run_dir / ("plots/cwola_eval/seed_" + str(seed))
        os.makedirs(plot_path, exist_ok=True)
        file_path=cfg.cwola_path+cfg.cwola_subfolders+f"standard/seed_{seed}/"+"cwola_outputs.h5"
        file_path_extra=cfg.cwola_path+cfg.cwola_subfolders+f"standard/seed_{seed}/"+"cwola_outputs_extra.h5"
        data = {}
        with pd.HDFStore(file_path, "r") as store:
                # Iterate over all the keys (dataset names) in the file
                for key in store:
                    # Read each dataset into a pandas DataFrame and store in the dictionary
                    data[key[1:]] = store[key]
                    
        data_extra = {}
        with pd.HDFStore(file_path_extra, "r") as store:
                # Iterate over all the keys (dataset names) in the file
                for key in store:
                    # Read each dataset into a pandas DataFrame and store in the dictionary
                    data_extra[key[1:]] = store[key]
        print(data)
        datasr = data["df"][data["df"]["CWoLa"] == 1]
        do_ROC(datasr["preds"], 
                datasr["is_signal"], 
                save_path=plot_path / "ROC")
        do_SI_v_rej(datasr["preds"], 
                    datasr["is_signal"], 
                    save_path=plot_path / "SI_v_rej")
        do_rejection_v_TPR(datasr["preds"], 
                            datasr["is_signal"], 
                            save_path=plot_path / "rejection_v_TPR")
        
        do_mass_sculpting(data["df"]["m_jj"], data["df"]["preds"], data["df"]["is_signal"], save_path=plot_path / "mass_sculpting.png")
        do_mass_sculpting(pd.concat([datasr["m_jj"], data_extra["df"]["m_jj"]]), 
                    pd.concat([datasr["preds"], data_extra["df"]["preds"]]), 
                    pd.concat([datasr["is_signal"], data_extra["df"]["m_jj"]*0]), save_path=plot_path / "mass_sculpting_density.png", density=True, rej_cuts = [0.9], bins=100)

    
def calc_TPR_FPR(scores, true_labels):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_labels = np.array(true_labels)[sorted_indices]

    # Initialize variables to calculate TPR and FPR
    TP = 0
    FP = 0
    FN = sum(true_labels)  # All positives initially
    TN = len(true_labels) - FN  # All negatives initially

    tpr_list = []
    fpr_list = []

    # Calculate TPR and FPR for each threshold
    for score, label in zip(sorted_scores, sorted_labels):
        if label == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        
        if TP + FN == 0:
            TPR = 0
        else:
            TPR = TP / (TP + FN)
        if FP + TN == 0:
            FPR = 0
        else:
            FPR = FP / (FP + TN)
        
        tpr_list.append(TPR)
        fpr_list.append(FPR)

    # Add (0,0) and (1,1) points to the ROC curve
    tpr_list.insert(0, 0.0)
    fpr_list.insert(0, 0.0)
    tpr_list.append(1.0)
    fpr_list.append(1.0)
    return tpr_list, fpr_list

def do_ROC(scores, true_labels, save_path, title="ROC", make_plot=True, save_npy=True):
    tpr_list, fpr_list = calc_TPR_FPR(scores, true_labels)

    # Plot the ROC curve
    if make_plot:
        plt.figure()
        plt.plot(fpr_list, tpr_list)
        plt.title(title)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.grid(which='major')
        plt.savefig(str(save_path)+".png", bbox_inches='tight', dpi=300)
    if save_npy:
        np.save(str(save_path)+".npy", np.array([fpr_list, tpr_list]))

def do_SI_v_rej(scores, true_labels, save_path, title="SI_v_rej", make_plot=True, save_npy=True):
    tpr_list, fpr_list = calc_TPR_FPR(scores, true_labels)
    SI = np.array(tpr_list) / np.sqrt(np.array(fpr_list))
    rej = 1 / np.array(fpr_list)
    # Plot the curve
    if make_plot:
        plt.figure()
        plt.plot(rej, SI)
        plt.title(title)
        plt.grid()
        plt.xscale('log')
        plt.xlabel('Rejection')
        plt.ylabel('SI')
        plt.grid(which='major')
        plt.savefig(str(save_path)+".png", bbox_inches='tight', dpi=300)
    if save_npy:
        np.save(str(save_path)+".npy", np.array([rej, SI]))

def do_rejection_v_TPR(scores, true_labels, save_path, title="rej_v_TPR", make_plot=True, save_npy=True):
    tpr_list, fpr_list = calc_TPR_FPR(scores, true_labels)
    rej = 1 / np.array(fpr_list)
    # Plot the curve
    if make_plot:
        plt.figure()
        plt.plot(tpr_list, rej)
        plt.title(title)
        plt.grid()
        plt.yscale('log')
        plt.xlabel('Rejection')
        plt.ylabel('SI')
        plt.grid(which='major')
        plt.savefig(str(save_path)+".png", bbox_inches='tight', dpi=300)
    if save_npy:
        np.save(str(save_path)+".npy", np.array([tpr_list, rej]))

def do_mass_sculpting(masses, scores, true_labels, save_path, rej_cuts = [0.5, 0.9, 0.95, 0.99], title="mass_sculpting", density=False, bins=100):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_masses = np.array(masses)[sorted_indices]
    sorted_scores = np.array(scores)[sorted_indices]
    sorted_true_labels = np.array(true_labels)[sorted_indices]
    
    plt.figure()
    _, bins, _ = plt.hist(sorted_masses, bins=bins, alpha=1, label="original", color='black', histtype='step', density=density)
    plt.hist(sorted_masses[sorted_true_labels==1], bins=bins, alpha=1, label="signal", color='gray', histtype='step', density=density)

    plt.yscale('log')
    plt.title(title)
    plt.xlabel('mass')
    plt.ylabel('counts')
    
    colormap = cm.get_cmap('jet', len(rej_cuts)+2)
    for i, rej_cut in enumerate(rej_cuts):
        cut_index = -int(len(sorted_scores) * rej_cut)
        plt.hist(sorted_masses[:cut_index], bins=bins, label=f"rejection {rej_cut}", color=colormap(i+1), histtype='step', density=density)
    plt.legend()
    plt.grid()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_cuts_mass_hists():
    pass

if __name__ == "__main__":
    main()
