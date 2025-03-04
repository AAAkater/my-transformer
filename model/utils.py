import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """
    位置编码
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout_rate: float = 0.1):
        super(PositionalEncoding, self).__init__()

        if d_model % 2 != 0:
            raise ValueError(f"{d_model}需为偶数!")
        self.dropout = nn.Dropout(p=dropout_rate)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(self.max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos.float() * div_term)
        pe[:, 1::2] = torch.cos(pos.float() * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        向前传播

        Args:
            x (Tensor): _description_

        Returns:
            Tensor: _description_
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)
