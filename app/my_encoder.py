import torch
import torch.nn.functional as F
from torch import Tensor, nn

from app.my_attention import MultiHeadSelfAttention, ResidualConnection, feed_forward


class TransformerEncoderLayer(nn.Module):
    """
    transformer编码器层
    """

    def __init__(
        self,
        d_model: int = 512,
        head_nums: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerEncoderLayer, self).__init__()
        query_dim = key_dim = value_dim = d_model // head_nums
        self.multi_head_self_attention = ResidualConnection(
            MultiHeadSelfAttention(
                d_model, query_dim, key_dim, value_dim, head_nums, dropout
            ),
            d_model,
            dropout,
        )

        self.feed_forward = ResidualConnection(
            feed_forward(d_model, dim_feedforward),
            d_model,
            dropout,
        )

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        向前传播

        Args:
            src (Tensor): _description_
            src_mask (Tensor, optional): _description_. Defaults to None.
        Returns:
            Tensor: _description_
        """
        src = self.multi_head_self_attention(src, src, src, src_mask)
        return self.feed_forward(src)
