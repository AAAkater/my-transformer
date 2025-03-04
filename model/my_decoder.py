import torch
import torch.nn.functional as F
from model.my_attention import MultiHeadSelfAttention, ResidualConnection, feed_forward

# from model.my_pos_encoding import PositionalEncoding
from torch import Tensor, nn


class TransformerDecoderLayer(nn.Module):
    """
    transformer解码器层
    """

    def __init__(
        self,
        d_model: int = 512,
        head_nums: int = 8,
        dim_feedforward: int = 2048,
        dropout_rate: float = 0.1,
    ):
        super(TransformerDecoderLayer, self).__init__()
        query_dim = key_dim = value_dim = max(d_model // head_nums, 1)
        self.self_attention = ResidualConnection(
            MultiHeadSelfAttention(
                d_model, query_dim, key_dim, value_dim, head_nums, dropout_rate
            ),
            d_model,
            dropout_rate,
        )

        self.cross_attention = ResidualConnection(
            MultiHeadSelfAttention(
                d_model, query_dim, key_dim, value_dim, head_nums, dropout_rate
            ),
            d_model,
            dropout_rate,
        )

        self.feed_forward = ResidualConnection(
            feed_forward(d_model, dim_feedforward),
            d_model,
            dropout_rate,
        )

    def forward(
        self,
        src: Tensor,
        memory: Tensor = None,
        src_mask: Tensor = None,
        memory_mask: Tensor = None,
    ) -> Tensor:
        """
        向前传播

        Args:
            src (Tensor): 目标序列张量
            memory (Tensor, optional): 编码器输出张量. Defaults to None.
            src_mask (Tensor, optional): 目标序列张量掩码. Defaults to None.
            memory_mask (Tensor, optional): 编码器输出张量掩码. Defaults to None.
        Returns:
            Tensor: 解码层输出
        """
        src = self.self_attention(src, src, src, src_mask)
        src = self.cross_attention(src, memory, memory, memory_mask)
        return self.feed_forward(src)


class TransformerDecoder(nn.Module):
    """
    transformer解码器
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        layer_count: int,
        d_model: int = 512,
        head_nums: int = 8,
        dim_feedforward: int = 2048,
        dropout_rate: float = 0.1,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    head_nums,
                    dim_feedforward,
                    dropout_rate,
                )
                for _ in range(layer_count)
            ]
        )

        self.final_linear = nn.Linear(d_model, d_model)

    def forward(
        self,
        src: Tensor,
        memory: Tensor = None,
        src_mask: Tensor = None,
        memory_mask: Tensor = None,
    ) -> Tensor:
        """
        向前传播

        Args:
            src (Tensor): 目标序列张量
            memory (Tensor, optional): 编码器输出张量. Defaults to None.
            src_mask (Tensor, optional): _description_. Defaults to None.
            memory_mask (Tensor, optional): _description_. Defaults to None.

        Returns:
            Tensor: 解码器输出
        """

        seq_len, dim_len = src.size(1), src.size(2)
        src += PositionalEncoding(dim_len)(src)
        for layer in self.layers:
            src = layer(src, memory, src_mask, memory_mask)

        return F.softmax(self.final_linear(src), dim=-1)
