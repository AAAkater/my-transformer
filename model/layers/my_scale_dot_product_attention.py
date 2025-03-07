import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        d_k = q.size(-1)

        # Q * K^T/sqrt(d_k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        attn = F.softmax(scores, dim=-1)

        return attn @ v, attn
