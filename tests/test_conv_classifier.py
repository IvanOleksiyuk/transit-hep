import multiprocessing
import tempfile
from functools import partial

import hydra
import lightning as L
import torch as T
from omegaconf import OmegaConf

from hyperproject.data.mnist import MNISTDataModule
from hyperproject.models.cnn import ConvClassifier


def test_network():
    digit_imgs = T.rand(10, 1, 28, 28)
    labels = T.rand(10, 10)

    model = ConvClassifier(
        (labels[:1], digit_imgs[:1]),
        conv_blocks_1=2,
        conv_blocks_2=2,
        hidden_conv_channels=8,
        mlp_depth=3,
        optimizer_factory=partial(T.optim.Adam, lr=1.0e-3),
    )
    labels_pred = model.network(digit_imgs).detach()
    assert labels_pred.shape == labels.shape
    assert T.all((labels_pred >= 0) & (labels_pred <= 1))

    sum_one_diff = T.abs(T.sum(labels_pred, dim=-1) - 1)
    print("sum_diff_mean =", T.mean(sum_one_diff))
    print("sum_diff_std =", T.std(sum_one_diff))
    assert T.mean(sum_one_diff) < 1e-4


def test_training():
    with tempfile.TemporaryDirectory() as load_path:
        dev_set_factory = hydra.utils.instantiate(
            OmegaConf.create(
                dict(
                    _target_="hyperproject.data.mnist.MNISTDataset",
                    _partial_=True,
                    load_path=load_path,
                    train=True,
                    size=10,
                )
            )
        )
        predict_set_factory = hydra.utils.instantiate(
            OmegaConf.create(
                dict(
                    _target_="hyperproject.data.mnist.MNISTDataset",
                    _partial_=True,
                    load_path=load_path,
                    train=False,
                    size=10,
                )
            )
        )

        num_workers = multiprocessing.cpu_count() - 1
        dev_loader_conf = OmegaConf.create(dict(batch_size=5, num_workers=num_workers))
        predict_loader_conf = OmegaConf.create(
            dict(batch_size=5, num_workers=num_workers)
        )

        datamod = MNISTDataModule(
            dev_set_factory=dev_set_factory,
            predict_set_factory=predict_set_factory,
            val_frac=0.5,
            dev_loader_conf=dev_loader_conf,
            predict_loader_conf=predict_loader_conf,
        )

    model = ConvClassifier(
        datamod.mock_sample(),
        conv_blocks_1=2,
        conv_blocks_2=2,
        hidden_conv_channels=8,
        mlp_depth=3,
        optimizer_factory=partial(T.optim.Adam, lr=1.0e-3),
    )

    with tempfile.TemporaryDirectory() as trainer_root_dir:
        trainer = L.Trainer(
            max_epochs=1,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=1,
            default_root_dir=trainer_root_dir,
        )
        trainer.fit(model, datamod)
        pred_batches = trainer.predict(model, datamod)

    labels_pred = datamod.invert_setup_on_prediction(pred_batches)["labels"]

    print("labels_pred.shape =", labels_pred.shape)
