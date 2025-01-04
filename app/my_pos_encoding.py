import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """
    位置编码
    """

    def __init__(self, dim: int, max_len: int = 5000, dropout_rate: float = 0.1):
        super(PositionalEncoding, self).__init__()

        if dim % 2 != 0:
            raise ValueError(f"{dim}需为偶数!")
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(pos.float() * div_term)
        pe[:, 1::2] = torch.cos(pos.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, emb: Tensor) -> Tensor:

        emb = emb + self.pe[: emb.size(0)]
        return self.dropout(emb)
