from torch import Tensor, nn

from models.transformer.layers.my_multi_head_attention import MultiHeadAttention
from models.transformer.layers.my_position_wise_feed_forward import (
    PositionWiseFeedForward,
)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
    ):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        enc: Tensor,
        mask: Tensor,
    ):
        residual = enc.clone()
        enc = self.attn(
            enc,
            enc,
            enc,
            mask,
        )
        # 残差连接
        enc += residual
        enc = self.norm1(enc)

        residual = enc.clone()
        enc = self.pos_ffn(enc)
        # 残差连接
        enc += residual
        enc = self.norm2(enc)

        return enc
