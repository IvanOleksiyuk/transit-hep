"""
A collection of plotting functions useful for jet utilities
"""

from copy import deepcopy
from itertools import combinations
from pathlib import Path, PosixPath
from typing import List, Union

import h5py
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import roc_auc_score, roc_curve

from jutils.data_loading import load_rodem_data


def kl_div(hist_1: np.ndarray, hist_2: np.ndarray):
    """Calculate the KL divergence between two binned densities"""
    assert hist_1.shape == hist_2.shape
    h_1 = hist_1 / np.sum(h_1)
    h_2 = hist_2 / np.sum(h_2)
    h_1 = np.clip(h_1, a_min=1e-8, a_max=None)  ## To make the logarithm safe!
    h_2 = np.clip(h_2, a_min=1e-8, a_max=None)
    return np.sum(h_1 * np.log(h_1 / h_2))


def js_div(hist_1: np.ndarray, hist_2: np.ndarray) -> float:
    """Calculate the Jensen-Shannon Divergence between two binned densities"""
    assert hist_1.shape == hist_2.shape
    M = 0.5*(hist_1+hist_2)
    return 0.5 * (kl_div(hist_1, M) + kl_div(hist_2, M))


def plot_roc_curves(
    output_path: str,
    networks: List[dict],
    processes: dict,
    br_at_eff: list = None,
    xlim: tuple = (0, 1),
    ylim: tuple = (1, 3e4),
    do_log: bool = True,
    fig_size: tuple = (6, 6),
    fig_format: str = "png",
):
    """
    Creates a collection of roc curves for each network.

    args:
        output_path: The save directory for the plots, will be created using mkdir.
        networks: A list of networks represented as dictionaries with 3 required keys.
            path: The location of the network's exported directory.
            label: The label for the legend.
            score_name: Name of the column in the HDF files to use as tagging score.
            other: Other keywords are passed to plt.plot (linestyle, color, ...).
        processes:
            A dict of processes to plot. Keys are used for label, items for filenames.
            All combinations of two processes are used in the rocs.
            Background is used first, Signal second.

    kwargs:
        br_at_eff:
            A list of sig eff values to show corresponding background rejection
            in the legend.
        fig_size: The size of the output figure.
        fig_format: The file format for the output figure; 'png', 'pdf', 'svg', etc.
    """
    print(f"Creating ROC curves for all combinations of: {list(processes.keys())}")
    networks = deepcopy(networks) ## To be safe with pops

    ## Create the output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    ## Get the padding length of for the legend entries
    label_pad_length = max([len(net["label"]) for net in networks]) + 1
    label_pad_length = max([label_pad_length, 7])  ## Allow for the Model title
    legend_header = f"{'Model':{label_pad_length}} {'AUC':>6}"
    if br_at_eff is not None:
        legend_header += "    ".join([""] + ["BR@" + str(val)[1:] for val in br_at_eff])

    ## Cycle through all combinations of using 2 processes
    for background, signal in list(combinations(processes, 2)):
        print(f" - {signal} vs {background}")

        ## Create the figure
        fig, axis = plt.subplots(1, 1, figsize=fig_size)

        ## Add the title entry for the legend
        axis.plot([], [], color="w", label=legend_header)

        ## Cycle through each of the networks
        for net in networks:

            ## Load and combine the two datasets for the roc curves
            scores = []  ## Will be read from the network's output file
            targets = []  ## Will be 0 for background and 1 for signal
            for targ, p in enumerate([background, signal]):
                with h5py.File(Path(net["path"]) / processes[p], "r") as f:
                    scores.append(f[net["score_name"]][:])
                    targets += len(f[net["score_name"]]) * [targ]
            scores = np.concatenate(scores).flatten()
            targets = np.array(targets)

            ## Calculate the inclusive ROC and the values for the ROC curve
            auc = roc_auc_score(targets, scores)
            fpr, tpr, _ = roc_curve(targets, scores)
            frr = np.divide(1, np.clip(fpr, 1e-8, None))

            ## Create the legend entry
            label = f"{net['label']:{label_pad_length}} {auc:.4f}"
            if br_at_eff is not None:
                fr_at_vals = [frr[np.abs(tpr - val).argmin()] for val in br_at_eff]
                label += "".join([f"{fr:>9.1f}" for fr in fr_at_vals])

            ## Add to the plot with the label
            plt_kwargs = net.copy()
            plt_kwargs.pop("path")
            plt_kwargs.pop("label")
            plt_kwargs.pop("score_name")
            axis.plot(tpr, frr, label=label, **plt_kwargs)

        ## Formatting and saving the inclusive roc curves
        axis.set_ylim(ylim)
        if do_log:
            axis.set_yscale("log")
        axis.set_ylabel(f"({background}) " + r"Backround Rejection $1/\epsilon_B$")

        axis.set_xlim(xlim)
        axis.set_xticks(np.linspace(0, 1, 11))
        axis.grid(which="both")
        axis.set_xlabel(f"({signal}) " + r"Signal Efficiency $\epsilon_S$")
        axis.legend(prop={"family": "monospace"}, loc=1)

        fig.tight_layout()
        outpath = Path(output_path) / f"roc_{signal}_vs_{background}"
        fig.savefig(outpath.with_suffix("." + fig_format))
        plt.close(fig)


def plot_score_distributions(
    output_path: str,
    network: dict,
    processes: dict,
    bins: np.ndarray,
    xlim: list = None,
    ylim: list = None,
    do_log: bool = True,
    score_scalling_fn: callable = None,
    fig_size: tuple = (4, 4),
    fig_format: str = "png",
) -> None:
    """
    Plot the histograms a single network score for each process

    args:
        output_path: The save directory for the plots, will be created using mkdir.
        networks: A dictionary describing the network with three required keys
            path: The location of the network's exported directory.
            label: The label for the x_axis.
            score_name: Name of the column in the HDF files to use as tagging score.
            other: Other keywords are passed to plt.plot (linestyle, color, ...).
        processes:
            A dict of processes to plot. Keys are used for labels, items for filenames.
        bins: The bins to use for the score histograms

    kwargs:
        xlim: The x limits of the plot
        ylim: The y limits of the plot
        do_log: If the y axis is logarithmic
        score_scalling_fn: Function to apply to the scores, useful for bounding
        fig_size: The size of the output figure.
        fig_format: The file format for the output figure; 'png', 'pdf', 'svg', etc.
    """
    print(
        f"Creating score distribution plots for {network['label']} on  {list(processes.keys())}"
    )

    ## Create the output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    ## Create the figure
    fig, axis = plt.subplots(1, 1, figsize=fig_size)

    ## For each of the processes
    for proc_name, file_name in processes.items():

        ## Load the scores from the distributions
        with h5py.File(Path(network["path"]) / file_name, "r") as f:
            scores = f[network["score_name"]][:].flatten()

            if score_scalling_fn is not None:
                scores = score_scalling_fn(scores)

        ## Create the histogram
        hist, edges = np.histogram(
            np.clip(scores, bins[0], bins[-1]), bins, density=True
        )

        ## Plot as a step function
        axis.step(edges, [0] + hist.tolist(), label=proc_name)

    ## Formatting axes
    if do_log:
        axis.set_yscale("log")
        axis.set_ylim(bottom=1e-1)
    else:
        axis.set_ylim(bottom=0)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_ylabel(f"Entries")
    axis.set_xlabel(f"{network['label']}({network['score_name']})")
    axis.legend()
    fig.tight_layout()

    ## Saving image
    outpath = Path(output_path) / (
        f"score_dist_{network['label']}({network['score_name']})"
    ).replace(" ", "_")
    fig.savefig(outpath.with_suffix("." + fig_format))
    plt.close(fig)


def plot_sculpted_distributions(
    output_path: str,
    network: dict,
    processes: dict,
    data_dir: str,
    bins: np.ndarray,
    br_values: np.ndarray,
    var: str = "mass",
    ylim: tuple = None,
    ratio_ylim: tuple = (0, 1),
    do_log: bool = True,
    do_norm: bool = True,
    redo_quantiles: bool = False,
    fig_size: tuple = (10, 4),
    fig_format: str = "png",
) -> np.ndarray:
    """
    Plot the histograms of the jet masses for different cuts

    The quantiles are taken using the background process which is the first process
    in the passed dictionary, unless redo_quantiles is True. Here the quantiles
    are recalculated per sample.

    args:
        output_path: The save directory for the plots, will be created using mkdir.
        networks: A dictionary describing the network with three required keys
            path: The location of the network's exported directory.
            label: The label for the x_axis.
            score_name: Name of the column in the HDF files to use as tagging score.
            other: Other keywords are passed to plt.plot (linestyle, color, ...).
        processes:
            A dict of processes to plot. Keys are used for label, items for filenames.
            Background process must be first!
        data_dir: The original directory of the data to pull the masses
        bins: The bins to use for mass histograms
        br_values: A list of background rejection values, each will be a hist
    kwargs:
        var: The jet variable to plot, must be either: pt, eta, phi, mass
        ylim: The y limits of the plots
        ratio_ylim: The y limits of the ratio plots
        do_log: If the y axis is logarithmic
        do_norm: If the histograms should be normalised
        redo_quantiles: Calculate the quantiles per sample or just background
        fig_size: The size of the output figure.
        fig_format: The file format for the output figure; 'png', 'pdf', 'svg', etc.
    """
    print(f"Creating {var} distributions after cuts on {network['label']} for {list(processes.keys())}")

    ## Create the output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    ## Create the figure
    if do_norm:
        fig, axes = plt.subplots(1, len(processes), figsize=fig_size)
        axes = np.array([axes])
    else:
        fig, axes = plt.subplots(
            2, len(processes), figsize=fig_size, gridspec_kw={"height_ratios": [3, 1]}
        )
    if len(processes) == 1:
        axes = axes.reshape(2-do_norm, -1)

    ## For each process in the dictionary
    thresholds = None
    var_idx = {"pt": 0, "eta": 1, "phi": 2, "mass": 3}[var]
    for i, (proc_name, proc_file) in enumerate(processes.items()):
        print(f" - loading {proc_name}")

        ## Load the jet var from the original HDF files: high lvl = pt, eta, phi, mass
        var_data = load_rodem_data(
            Path(data_dir) / proc_file,
            info="HL",
            n_cnsts=0,
            incl_substruc=False,
        )[:, var_idx]

        ## Load the scores
        with h5py.File(Path(network["path"]) / proc_file, "r") as f:
            scores = f[network["score_name"]][:].flatten()

        ## Check that they are of compatible length
        if len(var_data) != len(scores):
            raise ValueError(
                f"The scores in {proc_file} for {network['label']}",
                "do not match original data length",
            )

        ## Get the quantiles using the first process (background)
        if thresholds is None or redo_quantiles:
            thresholds = np.quantile(scores, br_values)

        ## Make a plot of the original function without any cuts
        orig_hist, edges = np.histogram(
            np.clip(var_data, bins[0], bins[-1]), bins, density=do_norm
        )
        axes[0, i].step(edges, [0] + orig_hist.tolist(), "k", label="Original")

        ## Add a dashed line at 1 on the ratio plot
        if not do_norm:
            axes[1, i].step([edges[0], edges[-1]], [1, 1], "--k")

        ## For each of the thresholds
        for thresh, br_val in zip(thresholds, br_values):

            ## Trim the data
            sel_data = var_data[scores >= thresh]

            ## Only histogram if there is more than one sample
            ## (sometimes nothing passes threshold)
            if len(sel_data) > 1:

                ## Create the histogram in the original plot
                hist, edges = np.histogram(
                    np.clip(sel_data, bins[0], bins[-1]), bins, density=do_norm
                )
                axes[0, i].step(edges, [0] + hist.tolist(), label=f"{br_val:.2f}")

                ## Create the ratio plot
                if not do_norm:
                    ratio = hist / (orig_hist + 1e-8)
                    axes[1, i].step(edges, [ratio[0]] + ratio.tolist())

            else:
                axes.plot(edges, [0] * len(edges), label=f"{br_val:.2f}")

        ## Formatting the histogram axis
        if do_log:
            axes[0, i].set_yscale("log")
        else:
            axes[0, i].set_ylim(bottom=0)
        axes[0, i].set_ylim(ylim)
        axes[0, i].set_xlim([edges[0], edges[-1]])
        axes[0, i].set_title(proc_name)
        axes[0, i].set_xlabel(var)

        ## Formatting the ratio plot
        if not do_norm:
            axes[0, i].set_xlabel("")
            axes[0, i].xaxis.set_ticklabels([])
            axes[1, i].set_ylim(ratio_ylim)
            axes[1, i].set_xlim((edges[0], edges[-1]))
            axes[1, i].set_xlabel(var)

        ## Remove y axis ticklabels from middle plots
        if i > 0:
            axes[0, i].yaxis.set_ticklabels([])
            if not do_norm:
                axes[1, i].yaxis.set_ticklabels([])

    ## Adjust and save the plot
    axes[0, 0].set_ylabel("a.u.")
    if not do_norm:
        axes[1, 0].set_ylabel("Ratio to Original")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    outpath = Path(
        output_path
    ) / f"{var}_dist_{network['label']}({network['score_name']})".replace(" ", "_")
    fig.savefig(outpath.with_suffix("." + fig_format))
    plt.close(fig)

    return np.array(thresholds)


def plot_mass_sculpting(
    output_path: str,
    networks: list,
    proc_name: str,
    proc_file: str,
    data_dir: str,
    bins: np.ndarray,
    br_values: np.ndarray,
    fig_format: str = "png",
    fig_size: tuple = (4, 4),
    ylim: tuple = None,
    xlim: tuple = None,
) -> None:
    """
    Plot the KLD to the original of the jet masses for different cuts

    args:
        output_path: The save directory for the plots, will be created using mkdir.
        networks: A list of dictionaries with three required keys
            path: The location of the network's exported directory.
            label: The label for the x_axis.
            score_name: Name of the column in the HDF files use as disciminant scores
            other: Other keywords are passed to plt.plot (linestyle, color, ...).
        proc_name: Name of the process to plot, used for saving filename
        proc_file: Filename of the process to load data in network dir and for mass
        data_dir: The original directory of the data to pull the masses
        bins: The bins to use for mass histograms
        br_values: A list of background rejection values to use for each histogram.

    kwargs:
        ylim: The y limits of the plot
        xlim: The x limits of the plot
        fig_size: The size of the output figure.
        fig_format: The file format for the output figure; 'png', 'pdf', 'svg', etc.
    """
    print(f"Creating mass sculpting plots for {proc_name}")
    networks = deepcopy(networks) ## To be safe with pops

    ## Create the output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    ## Create the figure
    fig, axis = plt.subplots(1, 1, figsize=fig_size)

    ## Load the masses from the original HDF files: high lvl = pt, eta, phi, mass
    print(" - Loading mass...")
    mass = load_rodem_data(
        Path(data_dir) / proc_file, info="HL", n_cnsts=0, incl_substruc=False
    )[:, -1]

    ## Get the original histogram of the masses
    orig_hist = np.histogram(mass, bins, density=True)[0]

    ## Cycle through the networks
    for network in networks:
        print(f" - {network['label']}")

        ## Load the scores
        with h5py.File(Path(network["path"]) / proc_file, "r") as f:
            scores = f[network["score_name"]][:].flatten()

        ## Check that they are of compatible length
        if len(mass) != len(scores):
            raise ValueError(
                f"The scores in {proc_file} for {network['label']}",
                "do not match original data length",
            )

        ## Get the thresholds for the score quantiles
        thresholds = np.quantile(scores, br_values)

        ## For each of these thresholds calculate the JD-divergence to the original
        jsd_vals = []
        for thresh in thresholds:
            sel_mass = mass[scores >= thresh]
            hist = np.histogram(sel_mass, bins, density=True)[0]
            jsd_vals.append(js_div(orig_hist, hist))

        ## Plot the data for the model
        network.pop("path")
        network.pop("score_name")
        plt.plot(br_values, jsd_vals, marker="o", **network)

    ## Formatting and saving image
    axis.set_ylim(bottom=0)
    axis.set_ylim(ylim)
    axis.set_xlim(xlim)
    axis.set_ylabel("Mass Sculpting")
    axis.set_xlabel("Background Rejection")
    axis.legend()

    fig.tight_layout()
    outpath = Path(output_path) / f"mass_sculpting_{proc_name}"
    fig.savefig(outpath.with_suffix("." + fig_format))
    plt.close(fig)


def plot_mass_score_correlation(
    output_path: str,
    network: dict,
    proc_name: str,
    proc_file: str,
    data_dir: str,
    mass_bins: np.ndarray,
    score_bins: np.ndarray,
    do_log: bool = False,
    score_scalling_fn: callable = None,
    cmap: str = "coolwarm",
    fig_format: str = "png",
    fig_size: tuple = (4, 4),
) -> None:
    """
    Plot and save a 2D heatmap showing the tagger's score verses the jet mass

    args:
        output_path: The save directory for the plots, will be created using mkdir.
        networks: A dictionary with three required keys
            path: The location of the network's exported directory.
            label: The label for the x_axis.
            score_name: Name of the column in the HDF files use as disciminant scores
            other: Other keywords are passed to plt.plot (linestyle, color, ...).
        proc_name: Name of the process to plot, used for saving filename
        proc_file: Filename of the process to load data in network dir and for mass
        data_dir: The original directory of the data to pull the masses
        mass_bins: The bins to use for mass in the 2d histogram
        score_bins: The bins to use for the scores in the 2d histogram

    kwargs:
        do_log: Use the log of the 2d histogram heights for the colours
        score_scalling_fn: Function to apply to the scores, useful for bounding
        cmap: The cmap to use in the heatmap
        fig_format: The file format for the output figure; 'png', 'pdf', 'svg', etc.
        fig_size: The size of the output figure.
    """
    print(f"Creating mass-score heatmap for {network['label']} on {proc_name}")

    ## Create the output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    ## Create the figure
    fig, axis = plt.subplots(1, 1, figsize=fig_size)

    ## Load the masses from the original HDF files: high lvl = pt, eta, phi, mass
    print(" - Loading mass...")
    mass = load_rodem_data(
        Path(data_dir) / proc_file, info="HL", n_cnsts=0, incl_substruc=False
    )[:, -1]

    ## Load the scores
    with h5py.File(Path(network["path"]) / proc_file, "r") as f:
        scores = f[network["score_name"]][:].flatten()

    ## Check that they are of compatible length
    if len(mass) != len(scores):
        raise ValueError(
            f"The scores in {proc_file} for {network['label']}",
            "do not match original data length",
        )

    ## Apply the scaling function to the scores
    if score_scalling_fn is not None:
        scores = score_scalling_fn(scores)

    ## Create the histogram and plot the heatmap
    hist, xedges, yedges = np.histogram2d(mass, scores, bins=[mass_bins, score_bins])
    imshow = axis.imshow(
        np.log(hist.T) if do_log else hist.T,
        origin="lower",
        cmap=cmap,
        extent=[min(xedges), max(xedges), min(yedges), max(yedges)],
        aspect="auto",
    )

    ## Include a colour bar
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(imshow, cax=cax, orientation="vertical")

    ## Formatting and saving image
    axis.set_ylabel(f"{network['label']}({network['score_name']})")
    axis.set_xlabel("Mass [GeV]")
    fig.tight_layout()
    outpath = Path(
        output_path
    ) / f"mass_score_heatmap_{proc_name}_{network['label']}({network['score_name']})".replace(
        " ", "_"
    )
    fig.savefig(outpath.with_suffix("." + fig_format))
    plt.close(fig)
