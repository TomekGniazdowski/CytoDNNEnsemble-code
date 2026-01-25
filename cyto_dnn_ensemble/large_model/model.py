import torch
from torch import flatten, nn

from cyto_dnn_ensemble.classification_heads import NNReprClassificationHead
from cyto_dnn_ensemble.model_utils import cnn_out_size


class CNN1DExtractorLarge(nn.Module):
    """1D Convolutional Neural Network for feature extraction."""
    
    def __init__(
        self, 
        num_cells: int, 
        num_markers: int
        ):
        
        super().__init__()
        
        self._representations_path = None

        self._cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_cells, out_channels=8, kernel_size=5, padding=1, stride=1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(True)
        )
        clf_in_dim = cnn_out_size(
            num_layers=4,
            in_size=num_markers,
            kernels_size_conv=[5, 3, 3, 3],
            stride_conv=[1, 1, 1, 1],
            padding_conv=[1, 1, 1, 1],
            kernel_size_pool=[2, 2, 2],
            stride_pool=[2, 2, 2]
        )

        self._clf = NNReprClassificationHead(in_dim=clf_in_dim * 64)

    def forward(self, x: torch.Tensor):
        z = self._cnn_layers(x)
        z = flatten(z, 1)
        if self._representations_path is not None:
            torch.save(z, self._representations_path)
        y = self._clf(z)
        return y


class LatentSpaceExtractorLarge(nn.Module):
    """Latent space extractor, based on pretrained CNN encoder."""
    
    def __init__(
        self, 
        cnn_encoder: nn.Module, 
        latent_dim: int, 
        num_markers: int,
        dataset_to_histograms_of_markers
        ):
        
        super().__init__()
        
        self.cnn_encoder = cnn_encoder
        self.latent_dim = latent_dim
        self.num_markers = num_markers
        self.dataset_to_histograms_of_markers = dataset_to_histograms_of_markers
        self._representations_path = None
        
        self._clf = NNReprClassificationHead(in_dim=latent_dim * num_markers)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        x = x.permute(0, 2, 1)
        x = self.dataset_to_histograms_of_markers.transform(x)
        z = self.cnn_encoder(x)
        z = z.view(batch_size, self.latent_dim * self.num_markers)
        if self._representations_path is not None:
            torch.save(z, self._representations_path)
        y = self._clf(z)
        return y