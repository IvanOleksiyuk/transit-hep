import tempfile

import hydra
import numpy as np
from omegaconf import OmegaConf

from hyperproject.data.mnist import MNISTDataModule


def test_setup():
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

        dev_loader_conf = OmegaConf.create(dict(batch_size=10))
        predict_loader_conf = OmegaConf.create(dict(batch_size=10))

        labels_pre = predict_set_factory()[:][0]

        datamod = MNISTDataModule(
            dev_set_factory=dev_set_factory,
            predict_set_factory=predict_set_factory,
            val_frac=0.1,
            dev_loader_conf=dev_loader_conf,
            predict_loader_conf=predict_loader_conf,
        )
        datamod.setup("fit")
        datamod.setup("val")
        datamod.setup("predict")

        dataloader = datamod.predict_dataloader()

        for _labels, _digit_imgs in dataloader:
            print("digit_imgs.shape:", _digit_imgs.shape)
            print("labels.shape:", _labels.shape)
            labels_post = datamod.invert_setup_on_prediction([_labels])["labels"]

    assert len(labels_pre) == 10
    assert len(labels_post) == 10
    assert np.all(labels_pre == labels_post)


def test_mock_sample():
    with tempfile.TemporaryDirectory() as load_path:
        dev_set_factory = hydra.utils.instantiate(
            OmegaConf.create(
                dict(
                    _target_="hyperproject.data.mnist.MNISTDataset",
                    _partial_=True,
                    load_path=load_path,
                    size=10,
                    train=True,
                )
            )
        )
        predict_set_factory = hydra.utils.instantiate(
            OmegaConf.create(
                dict(
                    _target_="hyperproject.data.mnist.MNISTDataset",
                    _partial_=True,
                    load_path=load_path,
                    size=10,
                    train=False,
                )
            )
        )

        dev_loader_conf = OmegaConf.create(dict(batch_size=1))
        predict_loader_conf = OmegaConf.create(dict(batch_size=1))

        datamod = MNISTDataModule(
            dev_set_factory=dev_set_factory,
            predict_set_factory=predict_set_factory,
            val_frac=0.1,
            dev_loader_conf=dev_loader_conf,
            predict_loader_conf=predict_loader_conf,
        )
        datamod.setup("fit")

        for labels, digit_imgs in datamod.train_dataloader():
            sample_real = labels, digit_imgs
        mock_sample = datamod.mock_sample()
        assert [sample_real[i].shape for i in range(len(sample_real))] == [
            mock_sample[i].shape for i in range(len(sample_real))
        ]
