import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F

"""Source: https://github.com/pwr-ai-representation-learning"""


class Autoencoder(LightningModule):
    """Base class for autoencoders."""
    
    def __init__(self, learning_rate: float, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        
        self.learning_rate = learning_rate
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x):
        return self._encoder(x)

    def reconstruct(self, x):
        return self._decoder(z=self(x))

    def training_step(self, batch, batch_size):
        x = batch.to(next(self._encoder.parameters()).dtype)
        xr = self.reconstruct(x)

        loss = F.mse_loss(input=xr, target=x)
        self.log("train/mse", loss.detach().item(), on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
