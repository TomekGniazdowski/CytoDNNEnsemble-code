import torch
from numpy.random import default_rng
from tqdm import tqdm


class StandardScalerPyTorch:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, dataset):
        self.mean_ = dataset.mean(dim=0, keepdim=True)
        self.std_ = dataset.std(dim=0, keepdim=True, unbiased=False)

    def transform(self, dataset):
        return (dataset - self.mean_) / self.std_


class DatasetToHistogramsOfMarkers: 
    def __init__(self, num_bins, seed):
        self.num_bins = num_bins
        
        self._rng = default_rng(seed=seed)
        self._scaler = StandardScalerPyTorch()
    
    def fit(self, dataset):
        org_shape = dataset.shape
        dataset = dataset.reshape(-1, org_shape[2])
        dataset = dataset[self._rng.choice(dataset.shape[0], int(len(dataset) * 0.1), replace=False)] # take 10% of the data to fit the scaler
        print("Histogram normalization (fit)")
        dataset = [torch.histc(r, bins=self.num_bins) for r in tqdm(dataset)]
        dataset = torch.stack(dataset)
        self._scaler.fit(dataset)
    
    def transform(self, dataset):
        org_shape = dataset.shape
        dataset = dataset.reshape(-1, org_shape[2])
        dataset = [torch.histc(r, bins=self.num_bins) for r in dataset]
        dataset = torch.stack(dataset)
        dataset = self._scaler.transform(dataset)
        return dataset

"""
import time
scaler = StandardScalerPyTorch()
data = torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.float32).cuda()
print(data)
scaler.fit(data)
print(scaler.mean_, scaler.std_)
print(scaler.transform(data))
print(scaler.transform(torch.tensor([[2, 2]]).cuda()))
"""

"""
dataset_cuda = torch.tensor(np.random.rand(256, 30, 1024), device='cuda').to(torch.double) # batch_size, num_markers, num_cells
dthom = DatasetToHistogramsOfMarkers(num_bins=100)
dthom.fit(dataset_cuda)
start = time.time()
print(dthom.transform(dataset_cuda).shape)
stop = time.time()
print(stop - start)
"""
