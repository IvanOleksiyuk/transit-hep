import argparse
from pathlib import Path

import numpy as np
from time import process_time_ns
import yaml

from mltools.mltools.plotting import plot_multi_correlations
from src.sampling import fit_and_sample_scanned
from src.template import build_template
from src.utils import FEAT_LATEX, get_preprocessor, load_data


def get_args():
    parser = argparse.ArgumentParser(description="Generate template using OLIWS")
    parser.add_argument(
        "--file_path",
        type=Path,
        default="/srv/beegfs/scratch/groups/rodem/LHCO/events_anomalydetection_v2.curtains.h5",
        help="The path to the LHCO file",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default="/srv/beegfs/scratch/users/l/leighm/linearresults/",
        help="The output folder",
    )
    parser.add_argument(
        "--num_signal",
        type=int,
        default=3000,
        help="The number of signal events to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed for the random number generator",
    )
    parser.add_argument(
        "--inpt_list",
        type=lambda x: x.split(","),
        default="m_j1,del_m,tau21_j1,tau21_j2,del_R",
        help="Comma seperated variable input list",
    )
    parser.add_argument(
        "--scanned_var",
        type=str,
        default="m_jj",
        help="The variable to scan over, defines the regions",
    )
    parser.add_argument(
        "--sampler_class",
        type=str,
        default="ATLASDIJETFit",
        help="The name of the sampler from samplers.py to use",
    )
    parser.add_argument(
        "--pdf_fit_on_all",
        action="store_true",
        help="Whether to fit the scanned pdf on all data or just the side bands",
    )
    parser.add_argument(
        "--signal_flag",
        type=str,
        default="is_signal",
        help="The name of the signal column in the dataframe",
    )
    parser.add_argument(
        "--sideband_1",
        type=lambda x: [int(x) for x in x.split("_")],
        default="3100_3300",
        help="Underscore seperated sideband 1 range",
    )
    parser.add_argument(
        "--sideband_2",
        type=lambda x: [int(x) for x in x.split("_")],
        default="3700_3900",
        help="Underscore seperated sideband 2 range",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="quantile",
        help="The type of preprocessor to use before calculating OT cost",
    )
    parser.add_argument(
        "--interp_preproc",
        action="store_true",
        help="Whether to interpolate in the preprocessed space",
    )
    parser.add_argument(
        "--plot_bands",
        action="store_true",
        help="Whether to plot the SB1, SR and SB2 regions (takes time)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3000,
        help="The batch size for the OT matching",
    )
    parser.add_argument(
        "--template_size",
        type=int,
        default=500_000,
        help="The number of samples to draw from the template",
    )
    parser.add_argument(
        "--autooversample",
        type=int,
        default=4,
        help="Sample the context n times the number of datapoints in sidebands.",
    )
    parser.add_argument(
        "--pure_sidebands",
        action="store_true",
        help="Prevent the sidebands from containing any signal.",
    )
    return parser.parse_args()


def setup_run() -> Path:
    """Print the arguments of the run and return the output path."""
    # Load the arguments
    args = get_args()

    # Create the output path
    dir0 = args.output_path
    dir1 = (
        f"window_{args.sideband_1[0]}_{args.sideband_1[1]}"
        f"__{args.sideband_2[0]}_{args.sideband_2[1]}"
    )
    dir2 = f"dope_{args.num_signal}"
    args.output_path = dir0 / dir1 / dir2
    args.output_path.mkdir(exist_ok=True, parents=True)

    # Combine the features lists
    args.features = [*args.inpt_list, args.scanned_var, args.signal_flag]

    # Print the arguments of the run
    print()
    print("=" * 80)
    print("Exporting template with OLIWS")
    print("-" * 80)
    print(f"[--] Output path: {args.output_path}")
    print(f"[--] Input list: {args.inpt_list}")
    print(f"[--] Scanned var: {args.scanned_var}")
    print(f"[--] Sideband 1: {args.sideband_1}")
    print(f"[--] Sideband 2: {args.sideband_2}")
    print(f"[--] Sampler class: {args.sampler_class}")
    print(f"[--] Seed: {args.seed}")
    print("-" * 80)

    return args


def main() -> None:
    print("[--] Setting up run")
    args = setup_run()

    print(f"[--] Loading data from {args.file_path}")
    time = process_time_ns()
    data, sb1, sb2, sr, extra_sig = load_data(
        args.file_path,
        args.inpt_list,
        args.scanned_var,
        args.signal_flag,
        args.sideband_1,
        args.sideband_2,
        args.num_signal,
        args.output_path,
        args.plot_bands,
        pure_sidebands=args.pure_sidebands,
    )
    data_load_time = process_time_ns() - time
    print(f"[--] Data loaded in {data_load_time/1e9:.2f} seconds")

    print(f"[--] Fitting and sampling pdf for {args.scanned_var}")
    time = process_time_ns()
    target_vars = fit_and_sample_scanned(
        data,
        sb1,
        sb2,
        args.output_path,
        (args.sideband_1[1], args.sideband_2[0]),  # Inbetween the SBs
        sampler_class=args.sampler_class,
        pdf_fit_on_all=args.pdf_fit_on_all,
        template_size=args.template_size if args.autooversample == 0 else args.autooversample * (len(sb1) + len(sb2)),
    )
    pdf_fit_time = process_time_ns() - time
    print(f"[--] PDF fitted and sampled in {pdf_fit_time/1e9:.2f} seconds")
    print(f"[--] Fitting preprocessor: {args.preprocessor}")
    preproc = get_preprocessor(args.preprocessor)
    preproc = preproc.fit(data[:, :-2])

    # Build the template using OT matching
    print("[--] building template")
    time = process_time_ns()
    template = build_template(
        sb1,
        sb2,
        target_vars,
        args.batch_size,
        preproc,
        seed=args.seed,
        interp_preproc=args.interp_preproc,
    )
    template_time = process_time_ns() - time
    print(f"[--] Template built in {template_time/1e9:.2f} seconds") 

    # Save the template
    print("-" * 80)
    print(f"[--] SR shape: {sr.shape}")
    print(f"[--] Template shape: {template.shape}")
    print(f"[--] Extra signal shape: {extra_sig.shape}")
    np.save(args.output_path / "sr.npy", sr)
    np.save(args.output_path / "template.npy", template)
    np.save(args.output_path / "extra_signal.npy", extra_sig)
    np.savetxt(args.output_path / "features.txt", np.array(args.features), fmt="%s")
    with open(args.output_path / "arguments.yaml", "w") as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)

    # Plot the interpolated data
    print("[--] plotting...")
    plot_multi_correlations(
        data_list=[sr[sr[:, -1] == 0, :-2], template[:, :-2]],
        data_labels=["SR", r"SB1 $\rightarrow$ SR $\cup$ SB2 $\rightarrow$ SR"],
        do_err=True,
        do_norm=True,
        hist_kwargs=[
            {"color": "tab:red", "ls": "-"},
            {"color": "tab:blue", "ls": "--"},
        ],
        legend_kwargs={
            "loc": "upper right",
            "frameon": False,
            "fontsize": 22,
            "bbox_to_anchor": (0.7, 0.95),
        },
        col_labels=[FEAT_LATEX[x] for x in args.inpt_list],
        n_bins=30,
        fig_scale=1,
        n_kde_points=15,
        path=args.output_path / "template.pdf",
    )

    template_time_dict = {
        "data_load_time": float(data_load_time / 1e9),
        "pdf_fit_time": float(pdf_fit_time / 1e9),
        "template_time": float(template_time / 1e9),
    }
    #Save as YAML file
    with open(args.output_path / "time.yaml", "w") as file:
        yaml.dump(template_time_dict, file)


if __name__ == "__main__":
    main()
