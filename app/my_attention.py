import math
from torch import Tensor, nn
import torch

import torch.nn.functional as F


def self_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = None,
    mask=None,
):
    d_k = query.size(-1)
    scores = torch.matmul(
        query,
        key.transpose(-2, -1),
    ) / math.sqrt(d_k)

    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    self_att = F.softmax(scores, -1)
    if dropout is not None:
        self_dropout = nn.Dropout(dropout)
        self_att = self_dropout(self_att)
    return torch.matmul(self_att, value), self_att


class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super(MultiHeadSelfAttention, self).__init__()

    def forward(
        self,
        head: int,
        d_model: int,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout=0.1,
        mask=None,
    ):
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn = None
        if mask is not None:
            mask = mask.unsqueese(1)

        n_batch = query.size(0)
        return
