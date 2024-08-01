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
import glob
from scipy.interpolate import interp1d
# TODO pyroot utils will remove the need for ../configs

def find_files_with_name(root_dir, filename):
    # Use glob to find all files matching the filename pattern
    search_pattern = os.path.join(root_dir, '**', filename)
    matching_files = glob.glob(search_pattern, recursive=True)
    return matching_files

def main(cfg) -> None:
    out_dir = Path(cfg.run_dir) / "plots/compare"
    out_dir.mkdir(exist_ok=True)
    curve_types = ["ROC", "SI_v_rej", "rejection_v_TPR"]
    
    methods=dict(cfg.methods)
    methods[cfg.main_method.name] = cfg.main_method.dir
    
    for curve_type in curve_types:    
        curves = {}
        for key, method_dir in methods.items():
            curves[key] = get_curve(method_dir, curve_type)
        print("curves loaded for ", curve_type)
        plot_curves(curves, curve_type, out_dir)
        
def filter_finite_values(x):
    return x[np.isfinite(x)]
    
def get_curve(method, curve_type):
    files = find_files_with_name(method, curve_type+".npy")
    if len(files) == 0:
        print("No files found for ", method,  curve_type)
    curves = []
    for file in files:
        curves.append(np.load(file))
    # Aggregate the curves that might have different x values (use interpolation)
    
    if len(curves) == 1:
        return [curves[0][0], curves[0][1], 0]
    # Sort curves so that x is always rising
    for curve in curves:
        if curve[0][0] > curve[0][-1]:
            curve[0] = curve[0][::-1]
            curve[1] = curve[1][::-1]
        else:
            curve[0] = curve[0]
            curve[1] = curve[1]
    
    # Find the largest x range that all curves have
    min_x = max([filter_finite_values(curve[0])[0] for curve in curves])
    max_x = min([filter_finite_values(curve[0])[-1] for curve in curves])
    
    # get a common x range
    common_x = np.sort(np.unique(np.concatenate([filter_finite_values(curve[0]) for curve in curves])))
    common_x = common_x[(common_x >= min_x) & (common_x <= max_x)]
    
    # Interpolate the y values for each curve
    interpolated_y = []
    for xy in curves:
        x = xy[0]
        y = xy[1]
        interp_func = interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))
        interpolated_y.append(interp_func(common_x))
    
    # Convert the list of arrays into a 2D array for easier computation
    interpolated_y = np.array(interpolated_y)

    # Calculate the mean and standard deviation at each x-value
    mean_y = np.mean(interpolated_y, axis=0)
    std_y = np.std(interpolated_y, axis=0)
            
    return [common_x, mean_y, std_y]

def plot_curves(curves, curve_type, out_dir):
    if curve_type == "ROC":
        plt.figure()
        for key, curve in curves.items():
            x, y, std = curve
            plt.fill_between(x, y-std, y+std, alpha=0.5)
            plt.plot(x, y, label=key)
        plt.title("ROC Curve")
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.legend()
        plt.grid(which='major', alpha=0.5)
        plt.savefig(str(out_dir)+"/ROC.png", bbox_inches='tight', dpi=300)
        print("ROC curve saved to ", str(out_dir)+"/ROC.png")
    if curve_type == "SI_v_rej":
        plt.figure()
        for key, curve in curves.items():
            x, y, std = curve
            plt.fill_between(x, y-std, y+std, alpha=0.5)
            plt.plot(x, y, label=key)
        plt.title("SI vs Rejection")
        plt.xlabel('Rejection')
        plt.ylabel('SI')
        plt.legend()
        plt.xscale('log')
        plt.ylim(0, 20.0)
        plt.grid(which='both', alpha=0.5)
        plt.savefig(str(out_dir)+"/SI_v_rej.png", bbox_inches='tight', dpi=300)
        print("SI vs Rejection curve saved to ", str(out_dir)+"/SI_v_rej.png")
    if curve_type == "rejection_v_TPR":
        plt.figure()
        for key, curve in curves.items():
            x, y, std = curve
            plt.fill_between(x, y-std, y+std, alpha=0.5)
            plt.plot(x, y, label=key)
        plt.title("Rejection vs TPR")
        plt.ylabel('Rejection')
        plt.xlabel('TPR')
        plt.legend()
        plt.yscale('log')
        plt.grid(which='both', alpha=0.5)
        plt.savefig(str(out_dir)+"/rejection_v_TPR.png", bbox_inches='tight', dpi=300)
        print("Rejection vs TPR curve saved to ", str(out_dir)+"/rejection_v_TPR.png")

if __name__ == "__main__":
    main()
