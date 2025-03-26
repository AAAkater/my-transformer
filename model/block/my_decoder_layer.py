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
        self.pos_ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        dec: Tensor,
        enc_out: Tensor,
        self_attn_mask,
        dec_enc_attn_mask,
    ):
        # self-attention
        residual = dec.clone()
        dec = self.self_attn(dec, dec, dec, self_attn_mask)
        dec = self.norm1(dec)
        dec += residual

        # encoder-decoder attention
        residual = dec.clone()
        dec = self.norm2(dec)
        dec, dec_enc_attn = self.enc_dec_attn(dec, enc_out, enc_out, dec_enc_attn_mask)
        dec += residual

        # position-wise feed-forward network
        residual = dec
        dec = self.norm3(dec)
        dec = self.pos_ffn(dec)
        dec += residual

        return dec, dec_enc_attn
