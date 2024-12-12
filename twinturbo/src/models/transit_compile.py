import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
import wandb

# Define the MLP module
class TRANSIT(LightningModule):
    def __init__(self, input_dim=5, **kwargs):
        super().__init__()
        # A simple two-layer MLP
        self.fc1 = nn.Linear(input_dim, 64)  # Combine x, y, z into one input
        self.fc2 = nn.Linear(64, input_dim)     # Output same dimension as x
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Concatenate inputs along the feature dimension
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        return output
    
    def training_step(self, batch, batch_idx):
        x, y, z = batch
        x_reco = self(x)
        loss = self.criterion(x, x_reco)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        x_reco = self(x)
        loss = self.criterion(x, x_reco)
        #self.log('valid/total_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
