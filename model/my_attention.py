import math

import numpy as np
import torch
import torch.nn.functional as F
from regex import T
from torch import Tensor, nn


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor = None,
    dropout: float = None,
):
    """
    缩放点积注意力机制

    Args:
        query (Tensor):Q
        key (Tensor): K
        value (Tensor): V
        mask (_type_, optional): 掩码. Defaults to None.
        dropout (_type_, optional): 随机丢弃. Defaults to None.

    Returns:
        _type_: _description_
    """
    d_k = query.size(-1)  # 查询最后一个向量的维度, 也就是d_k
    scores = torch.matmul(
        query,
        key.transpose(-2, -1),
    ) / math.sqrt(d_k)
    # 掩码处理
    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, float("-inf"))
    self_att = F.softmax(input=scores, dim=-1)
    # 随机丢弃一些注意力权重
    if dropout is not None:
        # Dropout 用于在训练过程中随机丢弃一些注意力权重，以防止过拟合
        _dropout = nn.Dropout(dropout)
        self_att: Tensor = _dropout(self_att)
    return torch.matmul(self_att, value)


class AttentionHead(nn.Module):
    """
    单头注意力机制
    """

    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
    ):
        super(AttentionHead, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Linear(d_model, d_k)
        self.key = nn.Linear(d_model, d_k)
        self.value = nn.Linear(d_model, d_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None):
        return scaled_dot_product_attention(
            self.query(query),
            self.key(key),
            self.value(value),
            mask,
            self.dropout,
        )


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制
    """

    def __init__(
        self, d_model: int, d_k: int, d_v: int, head_nums: int = 8, dropout: float = 0.1
    ):
        super(MultiHeadSelfAttention, self).__init__()

        self.heads = nn.ModuleList(
            [AttentionHead(d_model, d_k, d_v, dropout) for _ in range(head_nums)]
        )
        self.linear_out = nn.Linear(d_v * head_nums, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        """
        向前传播

        Args:
            query (Tensor): Q
            key (Tensor): K
            value (Tensor): V
            mask (Tensor, optional): 掩码. Defaults to None.

        Returns:
            Tensor: 多头自注意力机制的输出
        """
        head_output = [head(query, key, value, mask) for head in self.heads]
        return self.linear_out(torch.cat(tensors=head_output, dim=-1))


def feed_forward(input_dim: int = 512, inter_dim: int = 2048) -> nn.Sequential:
    """
    前馈网络

    Args:
        input_dim (int, optional): 输入层维度. Defaults to 512.
        inter_dim (int, optional): 中间层维度. Defaults to 2048.
    Returns:
        nn.Sequential: 前馈网络
    """
    return nn.Sequential(
        nn.Linear(input_dim, inter_dim),
        nn.ReLU(),
        nn.Linear(inter_dim, input_dim),
    )


class ResidualConnection(nn.Module):
    """
    残差连接
    """

    def __init__(self, sub_layer: nn.Module, dim: int, dropout: float = 0.1):
        super(ResidualConnection, self).__init__()
        # 子层可以是多头自注意力机制或者前馈网络
        self.sub_layer = sub_layer
        # 层归一化
        self.LayerNorm = nn.LayerNorm(dim)
        # 丢弃
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.LayerNorm(x + self.dropout(self.sub_layer(x)))


# class PositionWiseFeedForward(nn.Module):
#     """
#     FeedForward层
#     w2(
#         relu(w1(ayer_norm(x))+b1)
#         )+b2
#     Args:
#         nn (_type_): _description_
#     """

#     def __init__(self, d_model: int, d_ff: int, dropout: nn.Dropout):
#         super(PositionWiseFeedForward, self).__init__()
#         self.w_1 = nn.Linear(d_model, d_ff)
#         self.w_2 = nn.Linear(d_ff, d_model)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#         self.relu = nn.ReLU()
#         self.dropout_1 = dropout
#         self.dropout_2 = dropout

#     def forward(self, x: Tensor) -> Tensor:
#         inter: Tensor = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
#         outer: Tensor = self.dropout_2(self.w_2(inter))
#         return outer


# def subsequent_mask(size: int):
#     attn_shape = (1, size, size)

#     mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")

#     return (torch.from_numpy(mask) == 0).to(torch.bool)


# class Generator(nn.Module):
#     def __init__(self, d_model: int, vocab: int):
#         super().__init__()
#         self.linear = nn.Linear(d_model, vocab)
#         self.softmax = F.log_softmax(self.linear, dim=-1)

#     def forward(self, x: Tensor) -> Tensor:
#         return self.softmax(x)
