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
    run_dir = Path(cfg.general.run_dir)
    os.makedirs(run_dir/ "plots/cwola_eval", exist_ok=True)
    file_path = "/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/workspaces/adv1_gauss_corr_4_gap_twinturbo_usem_addgapmass/twinturbo_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS1024b/cwola/window_3100_3300__3700_3900/dope_3000/standard/seed_0/cwola_outputs.h5"
    file_path_extra = "/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/workspaces/adv1_gauss_corr_4_gap_twinturbo_usem_addgapmass/twinturbo_reco_cons0.01_smls0.001_adv3_LHCO_CURTAINS1024b/cwola/window_3100_3300__3700_3900/dope_3000/standard/seed_0/cwola_outputs_extra.h5"
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
    plot_ROC(data["df"]["preds"][data["df"]["CWoLa"] == 1], 
             data["df"]["is_signal"][data["df"]["CWoLa"] == 1], 
             save_path=run_dir / "plots/cwola_eval/ROC.png")
    plot_SI_v_rej(data["df"]["preds"][data["df"]["CWoLa"] == 1], 
                  data["df"]["is_signal"][data["df"]["CWoLa"] == 1], 
                  save_path=run_dir / "plots/cwola_eval/SI_v_rej.png")
    plot_rejection_v_TPR(data["df"]["preds"][data["df"]["CWoLa"] == 1], 
                         data["df"]["is_signal"][data["df"]["CWoLa"] == 1], 
                         save_path=run_dir / "plots/cwola_eval/rejection_v_TPR.png")
    
    mass_sculpting(data["df"]["m_jj"], data["df"]["preds"], data["df"]["is_signal"], save_path=run_dir / "plots/cwola_eval/mass_sculpting.png")
    mass_sculpting(pd.concat([data["df"]["m_jj"][data["df"]["CWoLa"] == 1], data_extra["df"]["m_jj"]]), 
                   pd.concat([data["df"]["preds"][data["df"]["CWoLa"] == 1], data_extra["df"]["preds"]]), 
                   pd.concat([data["df"]["is_signal"][data["df"]["CWoLa"] == 1], data_extra["df"]["m_jj"]*0]), save_path=run_dir / "plots/cwola_eval/mass_sculpting_density.png", density=True, rej_cuts = [0.9], bins=100)

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

def plot_ROC(scores, true_labels, save_path, title="ROC"):
    tpr_list, fpr_list = calc_TPR_FPR(scores, true_labels)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr_list, tpr_list)
    plt.title(title)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_SI_v_rej(scores, true_labels, save_path, title="SI_v_rej"):
    tpr_list, fpr_list = calc_TPR_FPR(scores, true_labels)
    SI = np.array(tpr_list) / np.sqrt(np.array(fpr_list))
    rej = 1 / np.array(fpr_list)
    # Plot the ROC curve
    plt.figure()
    plt.plot(rej, SI)
    plt.title(title)
    plt.grid()
    plt.xscale('log')
    plt.xlabel('Rejection')
    plt.ylabel('SI')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_rejection_v_TPR(scores, true_labels, save_path, title="rej_v_TPR"):
    tpr_list, fpr_list = calc_TPR_FPR(scores, true_labels)
    rej = 1 / np.array(fpr_list)
    # Plot the ROC curve
    plt.figure()
    plt.plot(tpr_list, rej)
    plt.title(title)
    plt.grid()
    plt.yscale('log')
    plt.xlabel('Rejection')
    plt.ylabel('SI')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def mass_sculpting(masses, scores, true_labels, save_path, rej_cuts = [0.5, 0.9, 0.95, 0.99], title="mass_sculpting", density=False, bins=100):
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
