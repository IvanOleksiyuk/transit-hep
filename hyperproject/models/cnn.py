from functools import partial
from typing import Tuple

import lightning as L
import torch as T
import torch.nn as nn


class ConvClassifier(L.LightningModule):
    DIGIT_COUNT = 10

    network: nn.Module
    optimizer_factory: partial

    def __init__(
        self,
        mock_sample: Tuple[T.Tensor, T.Tensor],
        conv_blocks_1: int,
        conv_blocks_2: int,
        hidden_conv_channels: int,
        mlp_depth: int,
        optimizer_factory: partial,
    ):
        super().__init__()
        # mock sample contains non-builtin types; ignore so we can load from checkpoint
        self.save_hyperparameters(ignore=("mock_sample",))

        _, digit_img_sample = mock_sample

        assert digit_img_sample.shape[-1] == digit_img_sample.shape[-2]
        img_size = digit_img_sample.shape[-1]

        assert conv_blocks_1 > 0 and conv_blocks_2 > 0
        img_size_after_conv_1 = img_size - 2 * conv_blocks_1
        img_size_after_conv_1 //= 2  # max pooling
        img_size_after_conv_2 = img_size_after_conv_1 - 2 * conv_blocks_2
        mlp_width = img_size_after_conv_2**2 * hidden_conv_channels

        self.network = nn.Sequential(
            nn.Conv2d(1, hidden_conv_channels, 3),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Conv2d(hidden_conv_channels, hidden_conv_channels, 3), nn.ReLU()
                )
                for _ in range(conv_blocks_1 - 1)
            ],
            nn.MaxPool2d(2),
            *[
                nn.Sequential(
                    nn.Conv2d(hidden_conv_channels, hidden_conv_channels, 3), nn.ReLU()
                )
                for _ in range(conv_blocks_2)
            ],
            nn.Flatten(),
            *[
                nn.Sequential(nn.Linear(mlp_width, mlp_width), nn.ReLU())
                for _ in range(mlp_depth)
            ],
            nn.Linear(mlp_width, self.DIGIT_COUNT),
            nn.Softmax(dim=-1),
        )

        self.optimizer_factory = optimizer_factory

    def configure_optimizers(self) -> T.optim.Optimizer:
        return self.optimizer_factory(self.network.parameters())

    def forward(self, digit_imgs: T.Tensor) -> T.Tensor:
        # digit_imgs must regard their values as one channel for convolution blocks;
        # also, we want to normalize them to [0, 1]
        digit_imgs = digit_imgs.unsqueeze(1) / 256
        return self.network(digit_imgs)

    def _development_step(
        self, batch: Tuple[T.Tensor, T.Tensor], batch_idx: int
    ) -> T.Tensor:
        labels, digit_imgs = batch
        labels_encoded = T.eye(self.DIGIT_COUNT, device=labels.device)[labels]

        labels_pred = self.forward(digit_imgs)
        return nn.functional.binary_cross_entropy(labels_pred, labels_encoded)

    def training_step(
        self, batch: Tuple[T.Tensor, T.Tensor], batch_idx: int
    ) -> T.Tensor:
        loss = self._development_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch: Tuple[T.Tensor, T.Tensor], batch_idx: int) -> None:
        loss = self._development_step(batch, batch_idx)
        self.log(
            "valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def predict_step(
        self, batch: Tuple[T.Tensor, T.Tensor], batch_idx: int
    ) -> T.Tensor:
        _, digit_imgs = batch
        return self.forward(digit_imgs)
