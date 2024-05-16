# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter

class OneHotEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int,  embedding_dim: Optional[int] = None, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None,
                 device=None, dtype=None):
        super().__init__(num_embeddings, num_embeddings, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, device, dtype)
        factory_kwargs = {'device': device, 'dtype': dtype} 

        weight = torch.eye(num_embeddings, **factory_kwargs)
        if embedding_dim:
            embedding_dim = max(num_embeddings, embedding_dim)
            weight = torch.zeros((num_embeddings, embedding_dim))
            eye = torch.eye(num_embeddings, **factory_kwargs)
            weight[:eye.shape[0], :eye.shape[1]] = eye

        self.weight = Parameter(weight, requires_grad=False)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, ind: int = None) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if ind is None:
            x = x + self.pe[:x.size(0)]
        else:
            x = x + self.pe[ind]
        return self.dropout(x)