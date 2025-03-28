from torch import Tensor, nn

from model.layers.my_multi_head_attention import MultiHeadAttention
from model.layers.my_position_wise_feed_forward import PositionWiseFeedForward


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
    ):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        dec: Tensor,
        enc_out: Tensor,
        self_attn_mask: Tensor,
        dec_enc_attn_mask: Tensor,
    ):
        # self-attention
        residual = dec.clone()
        dec = self.self_attn(
            dec,
            dec,
            dec,
            self_attn_mask,
        )
        # 残差连接
        dec += residual
        dec = self.norm1(dec)

        # encoder-decoder attention
        residual = dec.clone()
        dec = self.enc_dec_attn(
            dec,
            enc_out,
            enc_out,
            dec_enc_attn_mask,
        )
        # 残差连接
        dec += residual
        dec = self.norm2(dec)

        # position wise feed forward network
        residual = dec.clone()
        dec = self.ffn(dec)
        # 残差连接
        dec += residual
        dec = self.norm3(dec)

        return dec
