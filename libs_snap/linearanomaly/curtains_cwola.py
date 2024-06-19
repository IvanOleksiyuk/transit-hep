import argparse

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from utils import run_bdt_folds
import yaml

def get_args():
    parser = argparse.ArgumentParser(
        description="Perform anomaly detection using CWOLA"
    )

    # data paths and input
    parser.add_argument(
        "--file_path",
        type=Path,
        default="/srv/beegfs/scratch/groups/rodem/LHCO/events_anomalydetection_v2.curtains.h5",
        help="The path to the LHCO file",
    )
    parser.add_argument(
        "--curtains_path",
        type=Path,
        default="/srv/beegfs/scratch/users/s/senguptd/curtains/images/doped_lhco/run_0/",
        help="Path of the runs for curtains"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="standard",
        help="The mode to run in. Options include: standard and idealised",
    )

    parser.add_argument(
        "--inpt_list",
        type=lambda x: x.split(","),
        default="m_j1,del_m,tau21_j1,tau21_j2,del_R",
        help="Comma seperated variable input list",
    )
    parser.add_argument(
        "--signal_flag",
        type=str,
        default="is_signal",
        help="The name of the signal column in the dataframe",
    )
    parser.add_argument(
        "--scanned_var",
        type=str,
        default="m_jj",
        help="The variable to scan over, defines the regions",
    )

    # BDT args
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="The number of folds for the BDT",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=250,
        help="The maximum number of iterations for the BDT",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=1,
        help="Whether to use early stopping",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
        help="The fraction of data to use for validation",
    )
    parser.add_argument(
        "--class_weight",
        type=str,
        default="balanced",
        help="The class weight for the BDT",
    )
    parser.add_argument(
        "--num_ensemble",
        type=int,
        default=50,
        help="The number of classifiers to train",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        default="mean",
        help="The method to aggregate the ensemble",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed for the random number generator",
    )
    return parser.parse_args()

def main():

    args = get_args()

    input_list = args.inpt_list

    curtains_path = args.curtains_path
    with open(curtains_path/"exp_info.yml") as f:
        config = yaml.safe_load(f)
        num_signal = config["doping"]
        bins = config["bins"]
        bins = [float(bin) for bin in bins.split(",")]
        sideband1 = [bins[1],bins[2]]
        sideband2 = [bins[3],bins[4]]

    output_path = args.curtains_path/'bdt'
    save_path = output_path/f"{args.mode}/seed_{args.seed}"
    save_path.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("[--] Starting BDT CWoLa for CURTAINs")
    print("-"*80)
    print(f"[--] Running in {args.mode} mode")
    print(f"[--] Using {num_signal} signal events")
    print(f"[--] Running with seed {args.seed}")

    data = pd.read_hdf(args.file_path)
    data = data[[*input_list, args.scanned_var, args.signal_flag]]
    signal = data[data[args.signal_flag]]

    extra_signal = signal[int(num_signal):]
    extra_signal = extra_signal[extra_signal[args.scanned_var].between(sideband1[0], sideband2[1])]
    extra_signal = extra_signal.to_numpy().astype(np.float32)

    data = pd.concat([
        data[~data[args.signal_flag]],
        signal[:int(num_signal)],
    ])

    data = data.sample(frac=1).reset_index(drop=True)  # Above step made it bkg -> sig

    # Split the data into the two sidebands and the signal region
    data = data[data[args.scanned_var].between(sideband1[0], sideband2[1])]
    sr = data[data[args.scanned_var].between(sideband1[1], sideband2[0])]

    # Convert to numpy (faster)
    data = data.to_numpy().astype(np.float32)
    sr = sr.to_numpy().astype(np.float32)

    template = np.load(args.curtains_path/"evaluation/samples_sb1_2_to_sr.npz")
    template = [template[arr] for arr in ['arr_0', 'arr_1', 'arr_2', 'arr_3']]
    template = template[1] # This is the generated template.
    template = np.concatenate((template, -1*np.ones(len(template)).reshape(-1,1)), axis=1)



    # print(sr.shape, template.shape)
    print(f"[--] Template Shape: {template.shape}")
    print(f"[--] Data Shape: {sr.shape}")
    print(f"[--] Extra Signal Shape: {extra_signal.shape}")
    
    bdt_args_dict = {"max_iter": args.max_iter, "early_stopping": bool(args.early_stopping), "validation_fraction": args.validation_fraction, "class_weight": args.class_weight}

    final_df = run_bdt_folds(
        interp_data = template,
        sr = sr,
        extra_data = extra_signal,
        num_folds = args.num_folds,
        num_ensemble = args.num_ensemble,
        save_path = save_path,
        features = input_list,
        aggregate= args.aggregate,
        seed = args.seed,
        bdt_args = bdt_args_dict
    )

    # Template vs background score
    preds = final_df.preds.to_numpy()
    truth = final_df.Truth.to_numpy()
    bg_preds = preds[truth == 0]

    if args.mode.casefold() ==  "standard":
        template_preds = preds[truth == -1]
        
        closure_preds = np.concatenate((bg_preds, template_preds))
        closure_labels = np.concatenate((np.ones(len(bg_preds)),np.zeros(len(template_preds))))
        closure_fpr, closure_tpr, _ = roc_curve(closure_labels, closure_preds)
        closure_auc = auc(closure_fpr, closure_tpr)
        print("Closure AUC: ", closure_auc)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(closure_fpr, closure_tpr, label=f"Template AUC: {closure_auc:.3f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random")
        ax.legend(frameon=False, loc='upper left')
        fig.savefig(save_path/"closure_roc.png")

    if num_signal > 0:
        sig_preds = preds[truth == 1]
        perf_preds = np.concatenate((bg_preds, sig_preds))
        perf_labels = np.concatenate((np.zeros(len(bg_preds)), np.ones(len(sig_preds))))
        perf_fpr, perf_tpr, _ = roc_curve(perf_labels, perf_preds)
        perf_auc = auc(perf_fpr, perf_tpr)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(perf_fpr, perf_tpr, label=f"Performance AUC: {perf_auc:.3f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random")
        ax.legend(frameon=False, loc='upper left')
        fig.savefig(save_path/"svb_roc.png")

if __name__ == "__main__":
    os.environ["ZFIT_DISABLE_TF_WARNINGS"] = "1"
    main()
    print("All done. Exiting gracefully.")
