import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from cyto_dnn_ensemble.autoencoder_utils.utils import DatasetToHistogramsOfMarkers

"""Source: https://github.com/pwr-ai-representation-learning"""


class Dataset(LightningDataModule):
    """Creates lightning dataset."""

    def __init__(
        self,
        train_dataset_in: torch.Tensor,
        batch_size: int,
        num_bins: int,
        device: str,
        seed: int
        ):
        
        super().__init__()

        self.train_dataset_in = train_dataset_in
        self.batch_size = batch_size
        self.num_bins = num_bins
        
        self._train_X_in = torch.stack([
            self.train_dataset_in[i][0].permute(1, 0).float() for i in tqdm(range(len(self.train_dataset_in)))
            ]).to(device)
        
        self._dataset_to_histograms_of_markers = DatasetToHistogramsOfMarkers(num_bins=num_bins, seed=seed)
        self._train_ds = None

    def setup(self, stage=None):
        self._dataset_to_histograms_of_markers.fit(self._train_X_in)
        self._train_ds = self._dataset_to_histograms_of_markers.transform(self._train_X_in)

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

    def test_dataloader(self):
        return None
