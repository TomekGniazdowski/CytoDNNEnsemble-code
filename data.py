import pickle
import gc
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from torch import from_numpy
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils import generate_subsets, load_fcs_dataset, train_valid_split, combine_samples


class CytoDatasetFromCSV(Dataset):
    def __init__(
        self,
        csv_file,
        ncell,
        nsubset,
        per_sample=False
    ):
        """
        Args:
            - csv_file (str) :
                path to the .csv data file that contains the markers, sample_ids
                and phenotypes (labels). The columns of the .csv file is:
                `marker_1, marker_2, ..., marker_m, sample_id (int), label (str)`.
                Note: 1. the csv must have `sample_id` and `label` columns.
                      2. when using this csv dataloader, make sure the feature matrix is well
                         pre-processed and orgainized, e.g. marker selection and normalization.
            - ncell (int) :
                the number of cells per multi-cell input.
            - nsubset (int) :
                per_sample (bool): whether the `nsubset` argument refers to
                each class or each input sample.
        """
        samples = pd.read_csv(csv_file, index_col=0)
        id2pheno = pd.Series(samples['label'].values, index=samples['sample_id']).to_dict()
        sample_id = np.asarray(samples['sample_id']).astype(int)

        X_sample = samples.drop(columns=['sample_id', 'label']).to_numpy(dtype=np.float32)
        X_sample, sample_id = shuffle(X_sample, sample_id)

        if per_sample:
            self.data, self.label = generate_subsets(
                X_sample, id2pheno, sample_id, nsubset, ncell, per_sample
            )
        else:
            nsubset_list = []
            pheno_list = np.array([v for _, v in id2pheno.items()])
            for pheno in range(len(np.unique(pheno_list))):
                nsubset_list.append(nsubset // np.sum(pheno == pheno_list))
            self.data, self.label = generate_subsets(
                X_sample, id2pheno, sample_id, nsubset_list, ncell, False
            )
        self.data = np.transpose(self.data, (0, 2, 1)).astype(np.float16)
        self.label = self.label.astype(np.float16)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return from_numpy(self.data[idx]), self.label[idx]


class CytoDatasetFromFCS(Dataset):
    def __init__(
        self,
        X_sample,
        id_sample,
        phenotypes,
        ncell,
        nsubset,
        per_sample=False
    ):
        """
        Args:
            - X_sample (np.array: [total_cells x nmarks]) :
                the data matrix containing multiple samples
            - id_sample  (np.array: [total_cells]) :
                the sample id each cell belongs to
            - phenotypes (list: [number of samples]) :
                the `phenotype` each sample belongs to
        """
        phenotypes = np.asarray(phenotypes)
        if per_sample:
            self.data, self.label = generate_subsets(X_sample, phenotypes, id_sample, nsubset, ncell, per_sample)
        else:
            nsubset_list = []
            for pheno in range(len(np.unique(phenotypes))):
                nsubset_list.append(nsubset // np.sum(phenotypes == pheno))
            self.data, self.label = generate_subsets(X_sample, phenotypes, id_sample, nsubset_list, ncell, per_sample)
        self.data = np.transpose(self.data, (0, 2, 1)).astype(np.float16)
        self.label = self.label.astype(np.float16)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return from_numpy(self.data[idx]), self.label[idx]
    

def generete_train_val_test_data(
    args,
):
    
    if args.pkl:
        with open(args.train_pkl, 'rb') as f:
            _data = pickle.load(f)
            train_samples, train_phenotypes = _data['sample'], _data['phenotype']
        with open(args.test_pkl, 'rb') as f:
            _data = pickle.load(f)
            test_samples, test_phenotypes = _data['sample'], _data['phenotype']
    else:
        train_samples, train_phenotypes = load_fcs_dataset(
            args.train_fcs_info, args.markerfile, args.co_factor
        )
        test_samples, test_phenotypes = load_fcs_dataset(
            args.test_fcs_info, args.markerfile, args.co_factor
        )

    valid_phenotypes = train_phenotypes

    if (args.valid_fcs_info is not None or args.valid_pkl is not None) or args.generate_valid:
        if args.valid_fcs_info is not None:
            valid_samples, valid_phenotypes = load_fcs_dataset(
                args.valid_fcs_info, args.markerfile, args.co_factor
            )
            X_train, id_train = combine_samples(train_samples, np.arange(len(train_samples)))
            X_valid, id_valid = combine_samples(valid_samples, np.arange(len(valid_samples)))
        elif args.valid_pkl is not None:
            with open(args.valid_pkl, 'rb') as f:
                _valid = pickle.load(f)
            valid_samples, valid_phenotypes = _valid['sample'], _valid['phenotype']

            X_train, id_train = combine_samples(train_samples, np.arange(len(train_samples)))
            X_valid, id_valid = combine_samples(valid_samples, np.arange(len(valid_samples)))
        else:
            X_train, id_train, X_valid, id_valid = train_valid_split(
                train_samples, np.arange(len(train_samples))
            )
        del train_samples
        gc.collect()

        X_train, id_train = shuffle(X_train, id_train)
        train_data = CytoDatasetFromFCS(X_train, id_train, train_phenotypes,
                                        args.ncell, args.nsubset, args.per_sample)
        valid_data = CytoDatasetFromFCS(X_valid, id_valid, valid_phenotypes,
                                        args.ncell, args.nsubset, args.per_sample)
    else:
        X_train, id_train = combine_samples(train_samples, np.arange(len(train_samples)))
        del train_samples
        gc.collect()

        X_train, id_train = shuffle(X_train, id_train)
        train_data = CytoDatasetFromFCS(X_train, id_train, train_phenotypes,
                                        args.ncell, args.nsubset, args.per_sample)
        valid_data = CytoDatasetFromFCS(X_train, id_train, train_phenotypes,
                                        args.ncell, args.nsubset, args.per_sample)
    
    print("Classes distribution in train data:", dict(Counter([y for _, y in train_data])))
    print("Classes distribution in validation data:", dict(Counter([y for _, y in valid_data])))

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=4,
        drop_last=True,
        pin_memory=False
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=False,
        drop_last=False
    )
    
    return train_loader, valid_loader, train_data, valid_data, test_samples, test_phenotypes
