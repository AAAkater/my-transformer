from torch import Tensor, nn

from model.layers.my_multi_head_attention import MultiHeadAttention
from model.layers.my_position_wise_feed_forward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        enc: Tensor,
        self_attn_mask,
    ):
        # self-attention
        residual = enc.clone()
        enc = self.norm1(enc)
        enc, self_attn = self.self_attn(enc, enc, enc, self_attn_mask)
        enc += residual

        # position-wise feed-forward network
        residual = enc
        enc = self.norm2(enc)
        enc = self.pos_ffn(enc)
        enc += residual

        return enc, self_attn
