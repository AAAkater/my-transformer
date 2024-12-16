import math
from torch import Tensor, nn
import torch

import torch.nn.functional as F


def self_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: nn.Dropout = None,
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
        self_att = dropout(self_att)
    return torch.matmul(self_att, value), self_att


class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super(MultiHeadSelfAttention, self).__init__()

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        head: int = 8,
        d_model: int = 512,
        dropout=0.1,
        mask=None,
    ):
        """

        Args:
            query (Tensor):Q
            key (Tensor): K
            value (Tensor): V
            head (int, optional): 头数. Defaults to 8.
            d_model (int, optional): 输入维度. Defaults to 512.
            dropout (float, optional): _description_. Defaults to 0.1.
            mask (_type_, optional): _description_. Defaults to None.
        """
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

        query = (
            self.linear_query(query)
            .view(n_batch, -1, self.head, self.d_k)
            .transpose(1, 2)
        )
        key = (
            self.linear_value(key)
            .view(n_batch, -1, self.head, self.d_k)
            .transpose(1, 2)
        )
        value = (
            self.linear_value(value)
            .view(n_batch, -1, self.head, self.d_k)
            .transpose(1, 2)
        )

        x, self.attn = self_attention(
            query,
            key,
            value,
            dropout=self.dropout,
            mask=mask,
        )

        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head + self.d_k)

        return self.linear_out(x)
