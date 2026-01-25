import torch
from torch import nn


class NNReprClassificationHead(nn.Module):
    """Initialize the representation classification model."""
    
    def __init__(
        self, 
        in_dim: int
    ):

        super().__init__()

        self._clf = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.LeakyReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._clf(x)
        x = torch.squeeze(x, 1)
        return x


class NNFinalHeadClf(nn.Module):
    """Initialize the final classification model as classifier."""
    
    def __init__(
        self,
        hidden_dim: int
    ):

        super().__init__()

        self._clf = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LeakyReLU(True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._clf(x)
        x = torch.squeeze(x, 1)
        return x


class NNFinalHeadAttention(nn.Module):
    """Initialize the final classification model as simple attention."""
    
    def __init__(
        self
    ):
        
        super().__init__()
        
        self._clf = nn.Sequential(
            nn.Linear(2, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._clf(x)