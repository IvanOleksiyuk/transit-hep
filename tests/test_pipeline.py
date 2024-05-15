import os
import subprocess as sp
import tempfile


def test_pipeline():
    with tempfile.TemporaryDirectory() as workdir:
        sp.check_call(
            [
                "python",
                "scripts/train.py",
                "trainer.max_epochs=1",
                "trainer.logger=null",  # we don't want to clog wandb from unit tests
                "+datamodule.dev_set_factory.size=1_000",
                "+datamodule.predict_set_factory.size=1_000",
                "datamodule.dev_loader_conf.batch_size=512",
                "datamodule.predict_loader_conf.batch_size=512",
                f"io.dataset_path={workdir}",
                f"io.checkpoints_path={workdir}/checkpoints",
                f"io.result_path={workdir}/result.ckpt",
                f"io.trainer_root={workdir}",
                f"io.logging_dir={workdir}",
                "common=default",  # we don't have a private config on gitlab
                f"hydra.run.dir={workdir}/hydra/train",
            ]
        )
        assert os.path.isfile(f"{workdir}/result.ckpt")

        sp.check_call(
            [
                "python",
                "scripts/predict.py",
                f"io.checkpoint_path={workdir}/result.ckpt",
                f"io.predictions_save_path={workdir}/prediction.h5",
                "common=default",
                f"hydra.run.dir={workdir}/hydra/predict",
            ]
        )
        assert os.path.isfile(f"{workdir}/prediction.h5")

        sp.check_call(
            [
                "python",
                "scripts/plot.py",
                f"io.checkpoint_path={workdir}/result.ckpt",
                f"+io.prediction_paths.main={workdir}/prediction.h5",
                f"io.output_path={workdir}/plots.pdf",
                "common=default",
                f"hydra.run.dir={workdir}/hydra/plot",
            ]
        )
        assert os.path.isfile(f"{workdir}/plots.pdf")
