import rootutils

from mltools.mltools.utils import standard_job_array

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


def main() -> None:
    """Main executable script."""
    standard_job_array(
        job_name="anomaly",
        work_dir=root,
        log_dir=root / "logs",
        image_path="/srv/fast/share/rodem/images/diffbeit-image_latest.sif",
        command="python run.py",
        n_gpus=0,
        n_cpus=6,
        time_hrs=1,
        mem_gb=16,
        opt_dict={
            "num_signal": [
                250,
                500,
                1000,
                1500,
                2000,
                2500,
                3000,
            ],
            "seed": [0, 1, 2, 3, 4],
        },
        use_dashes=True,
        is_grid=True,
    )


if __name__ == "__main__":
    main()
