import math
from torch import Tensor, nn
import torch


class PositionalEncoding(nn.Module):
    """
    位置编码

    Args:
        nn (_type_): _description_
    """

    def __init__(self, dim: int, dropout: nn.Dropout, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        if dim % 2 != 0:
            raise ValueError(f"{dim}需为偶数!")
        self.dropout = dropout
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

    def forward(self, emb: Tensor):

        emb = emb + self.pe[: emb.size(0)]
        return self.dropout(emb)
