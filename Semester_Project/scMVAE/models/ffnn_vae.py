from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .vae import ModelVAE
from ...data.vae_dataset import VaeDataset
from ..components import Component

class FeedForwardVAE(ModelVAE):

    def __init__(self, h_dim: int, components: List[Component], dataset: VaeDataset,
                 scalar_parametrization: bool) -> None:
        super().__init__(h_dim, components, dataset, scalar_parametrization)
     
        #data dimensions
        self.in_dim = dataset.in_dim
        self.h_dim = h_dim
        #empty tensor 
        self.batch_saver = None
        #batch data
        self.batch_data = dataset.get_batch_effect()
        self.batch_data_dim = self.batch_data.shape[1]
        # 1 hidden layer encoder
        self.fc_e0 = nn.Linear(dataset.in_dim , h_dim)
        # 1 hidden layer decoder
        self.fc_d0 = nn.Linear(self.total_z_dim + self.batch_data_dim, h_dim)
        self.fc_logits = nn.Linear(h_dim, dataset.in_dim) 
        # Batch layer for normailzation
        self.batch_norm = nn.BatchNorm1d(self.h_dim)
        self.batch_norm_decoder = nn.BatchNorm1d(self.h_dim)

    def encode(self, x: Tensor) -> Tensor:
        """Encodes the Tensor `x` and saves the batch data in self.batch_saver.

        Args:
            x (Tensor): Tensor to encode.

        Returns:
            Tensor: Encoded tensor.
        """
        assert len(x.shape) == 2
        bs, dim = x.shape
        assert dim == self.in_dim
        x = x.view(bs, self.in_dim)
        #save the batch effect
        self.batch_saver = x[:,-self.batch_data_dim:]
        #forward pass
        x = torch.relu(self.batch_norm(self.fc_e0(x)))
        return x.view(bs, -1)

    def decode(self, concat_z: Tensor) -> Tensor:
        """Decodes the latent Tensor z and reconstructs the Tensor x, while taking into account the batch effect.
        Args:
            concat_z (Tensor): latent vector z.

        Returns:
            Tensor: Reconstructed tensor.
        """

        assert len(concat_z.shape) >= 2  
        bs = concat_z.size(-2)
        #concat the batch effect to latent space     
        if len(concat_z.shape) == 2:
            concat_z = torch.cat([concat_z, self.batch_saver], dim=1)
            x = torch.relu(self.batch_norm_decoder(self.fc_d0(concat_z)))
            x = self.fc_logits(x)
            x = x.view(-1, bs, self.in_dim)
        elif len(concat_z.shape) == 3:
            self.batch_saver = self.batch_saver.expand(500,self.batch_saver.shape[0],self.batch_saver.shape[1])
            concat_z = torch.cat([concat_z, self.batch_saver], 2)
            x = torch.relu(self.fc_d0(concat_z))
            x = self.fc_logits(x)
            x = x.view(-1,bs, self.in_dim)
        else:
            assert 0, "Not a Tensor"
        return x.squeeze(dim=0)




