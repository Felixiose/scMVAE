from typing import Tuple, Optional
import pandas as pd
import numpy as np
from scipy.io import mmread
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from Semester_Project.data.vae_dataset import VaeDataset
from torch.distributions import NegativeBinomial


class scRNADataset(VaeDataset):
    """
    Load a dataset using this class:
        data_file: path to a matrix data file
        label_file: path to a label data file
        batch_files: list of batch effect file(s) 
    1st dimension (number of rows) in data_file and batch file(s) must match
    1st dimension (number of rows) in data_file and label file must match
    batch data available as self.batch_data
    batch data dimensions as self.batch_data_dim
    """

    def __init__(self, batch_size: int, data_folder: str, data_file: str,
                 label_file: str, batch_files: Optional[list] = None,
                 doubles=False) -> None:
        
        # input file paths
        self.data_file = data_file
        self.batch_files = batch_files
        self.label_file = label_file
        self.data_folder = data_folder
        self.doubles = doubles
        
        # read batch effect, labels, data
        print("Reading data from " + self.data_file)
        if self.batch_files is not None and len(self.batch_files) > 0:
            self.batch_data_pd = self._read_batcheff()
        else:
            self.batch_data_pd = None

        self.data = self._read_data()
        self.batch_data = self.df_to_tensor(self.batch_data_pd)
        self.batch_data_dim = self.batch_data.shape[1]
        self.labels, self.label_names = self._read_label()
        self._max_data = self.get_max_data()

        self.check_dim1()
        self.dataset = torch.utils.data.TensorDataset(self.data, self.labels)
        self.dataset_len = self.__len__()
        self._in_dim = self.dataset.tensors[0].shape[1]
        self._shuffle_split_indx()
       
        # remove data, labels, batch_data_pd
        self.data = None
        self.labels = None
        self.label_names = None
        self.batch_data_pd = None

        super().__init__(batch_size, img_dims=None, in_dim=self._in_dim)  # FIXME: correct dimensions? -Colin

    def __len__(self) -> int:
        return self.dataset.tensors[0].size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates one sample
        """
        data, labels = self.dataset[idx], self.labels[idx]
        return torch.Tensor(data), torch.Tensor(labels)

    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
            return device

    def df_to_tensor(self, dataframe):
        device = self.get_device()
        if self.doubles:
            res = torch.from_numpy(dataframe.values).double().to(device)
        else:
            res = torch.from_numpy(dataframe.values).float().to(device)
        return res

    def read_mtx(self, filename, dtype='int32'):
        x = mmread(filename).astype(dtype)
        x = x.transpose().todense()
        return x

    def get_batch_effect(self):
        return self.batch_data

    # read batch effect file(s)
    def _read_batcheff(self):
        if len(self.batch_files) == 1:
            batch_data, batch_names = self._read_factorize_data(self.batch_files[0])
        else:
            list_batch = list()
            for one_file in self.batch_files:
                one_file_batch_data, one_file_batch_names = self._read_factorize_data(one_file)
                list_batch.append(one_file_batch_data)
            batch_data = pd.concat(list_batch, axis=1)
        return batch_data

    def _read_factorize_data(self, filepath):
        """
        read a file with a single column, factorize to numerical classes 
        return factorized data and names as pd.DataFrames
        """
        names = pd.read_csv(filepath, header=None)
        fct = pd.DataFrame(pd.factorize(names[0])[0]+1)

        return fct, names

    def _read_label(self):
        label_fct, label_names = self._read_factorize_data(self.label_file)
        label_fct = self.df_to_tensor(label_fct)
        return label_fct, label_names

    # read data and batch effect files
    def _read_data(self):
        """
        Read gene expression data (features), normalize by max value, append batch effect features
        """
        data = self.read_mtx(self.data_file)
        data = pd.DataFrame(data)
        
        # add constant dummy batch effect if no batch effect given
        if self.batch_data_pd is None:
            self.batch_data_pd = self.create_dummy_batch_eff(n=data.shape[0])
        
        data = pd.DataFrame(pd.concat([data, self.batch_data_pd], axis=1))
        res = self.df_to_tensor(data)
        return res

    def get_max_data(self):
        data = self.read_mtx(self.data_file)
        return data.max()

    def create_dummy_batch_eff(self, n: int):
        dummy = np.full(shape=(n, 1),
                        fill_value=0,
                        dtype='int32')
        dummy = pd.DataFrame(dummy)
        return dummy

    def check_dim1(self):
        assert (self.batch_data.shape[0] == self.data.shape[0]
                ), "batch_data.shape: %s, data.shape: %s" % (
            self.batch_data.shape[0], self.data.shape[0],
        )
        assert (self.labels.shape[0] == self.data.shape[0]
                ), "labels.shape: %s, data.shape: %s" % (
            self.labels.shape[0], self.data.shape[0],
        )

    def _shuffle_split_indx(self):
        indices = list(range(self.__len__()))
        split = int(np.floor(0.5 * self.__len__()))
        np.random.shuffle(indices)
        self.train_sampler = SubsetRandomSampler(indices[:split])
        self.test_sampler = SubsetRandomSampler(indices[split:])

    def create_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            sampler=self.train_sampler,
        )
        test_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            sampler=self.test_sampler,
        )
        return train_loader, test_loader
    
    def create_sequential_loader(self, batch_size: int) -> Tuple[DataLoader, DataLoader]:
        seq_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
        )
        return seq_loader

    def reconstruction_loss(self, x_mb_: torch.Tensor, x_mb: torch.Tensor) -> torch.Tensor:
        x_mb_nonnegative = torch.add(torch.nn.functional.relu(x_mb_), 0.00001)
        log_prob = -NegativeBinomial(x_mb_nonnegative, 0.5 * torch.ones_like(x_mb_)).log_prob(x_mb)
        scale_penalty = 1
        error = torch.sub(x_mb_, x_mb)
        penatly_term = torch.sqrt(scale_penalty * (error * error))
        return torch.add(log_prob, penatly_term)
