from torch import nn

from cyto_dnn_ensemble.autoencoder_utils.autoencoder import Autoencoder
from cyto_dnn_ensemble.model_utils import cnn_out_size

"""Source: https://github.com/pwr-ai-representation-learning"""


class CNNEncoderSmall(nn.Module):
    """Convolutional encoder for 1D signals."""
    
    def __init__(self, latent_dim: int, num_bins: int):
        super().__init__()

        conv_out_size = cnn_out_size(
            num_layers=2,
            in_size=num_bins,
            kernels_size_conv=[5, 3],
            stride_conv=[2, 2],
            padding_conv=[1, 1],
            kernel_size_pool=None,
            stride_pool=None,
        )
        
        self._layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, padding=1, stride=2),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(True),
            
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True),
            
            nn.Flatten(),
            nn.Linear(in_features=conv_out_size * 8, out_features=latent_dim),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self._layers(x)
        return x


class CNNDecoderSmall(nn.Module):
    """Convolutional decoder for 1D signals."""
    
    def __init__(self, latent_dim: int, num_bins: int):
        super().__init__()

        conv_out_size = cnn_out_size(
            num_layers=2,
            in_size=num_bins,
            kernels_size_conv=[5, 3],
            stride_conv=[2, 2],
            padding_conv=[1, 1],
            kernel_size_pool=None,
            stride_pool=None,
        )
        
        self._layers = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=conv_out_size * 8),
            nn.Unflatten(dim=1, unflattened_size=(8, conv_out_size)),
            
            nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=3, padding=1, stride=2, output_padding=0),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(True),
            
            nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size=5, padding=1, stride=2, output_padding=1)
        )

    def forward(self, z):
        x = self._layers(z)
        x = x.squeeze(1)
        return x


class CNNAutoencoderSmall(Autoencoder):
    """Convolutional autoencoder for 1D signals."""
    
    def __init__(self, learning_rate, latent_dim: int, num_bins: int):
        super().__init__(
            learning_rate=learning_rate,
            encoder=CNNEncoderSmall(latent_dim=latent_dim, num_bins=num_bins),
            decoder=CNNDecoderSmall(latent_dim=latent_dim, num_bins=num_bins),
        )