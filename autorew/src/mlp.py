import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn


# Define the MLP model with several hidden layers using PyTorch Lightning
class MLPClassifier(pl.LightningModule):
    def __init__(
        self,
        inpt_dim,
        hidden_sizes,
        num_classes,
        optimizer,
        scheduler,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        layers = []
        current_size = inpt_dim["data"][0]+inpt_dim["cond"][0]
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, num_classes))
        self.model = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, mode="train"):
        inputs = batch["data"].to(dtype=torch.float32)
        cond = batch["cond"].to(dtype=torch.float32)
        labels = batch["label"].squeeze()
        outputs = self(torch.cat([inputs, cond], dim=1))
        loss = self.criterion(outputs, labels)
        self.log(f"{mode}/total_loss", loss)

        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = self.accuracy(preds, labels)
        self.log(f"{mode}/accuracy", acc, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, mode="valid")

    def configure_optimizers(self) -> dict:
        """Configure the optimisers and learning rate sheduler for this
        model.
        """
        # Finish initialising the partialy created methods
        opt = self.hparams.optimizer(params=self.parameters())

        sched = self.hparams.scheduler.scheduler(opt)

        # Return the dict for the lightning trainer
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, **self.hparams.scheduler.lightning},
        }

    # Concept:

    # def on_fit_start(self, *_args):
    #    pass

    # def on_validation_epoch_end(self):
    #    pass


def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0):
    """Predict step for the model."""
    inputs = batch["data"].to(dtype=torch.float32)
    cond = batch["cond"].to(dtype=torch.float32)
    outputs = self(torch.cat([inputs, cond], dim=1))
    return outputs
