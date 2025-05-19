from torch import Tensor, nn

from models.transformer.layers.my_multi_head_attention import MultiHeadAttention
from models.transformer.layers.my_position_wise_feed_forward import (
    PositionWiseFeedForward,
)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
    ):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_head)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        dec_out: Tensor,
        enc_out: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ):
        """

        Args:
            dec_out (Tensor): 解码器输出
            enc_out (Tensor): 编码器输出
            src_mask (Tensor): 来自编码器的mask
            tgt_mask (Tensor): 来自解码器的mask

        Returns:
            _type_: 解码器输出
        """
        # self-attention
        residual = dec_out.clone()
        dec_out = self.dec_self_attn(
            dec_out,
            dec_out,
            dec_out,
            tgt_mask,
        )
        # 残差连接
        dec_out += residual
        dec_out = self.norm1(dec_out)

        # encoder-decoder attention
        residual = dec_out.clone()
        dec_out = self.enc_dec_attn(
            dec_out,
            enc_out,
            enc_out,
            src_mask,
        )
        # 残差连接
        dec_out += residual
        dec_out = self.norm2(dec_out)

        # position wise feed forward network
        residual = dec_out.clone()
        dec_out = self.ffn(dec_out)
        # 残差连接
        dec_out += residual
        dec_out = self.norm3(dec_out)

        return dec_out
