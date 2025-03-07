from torch import Tensor, nn

from model.block.my_decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_head: int,
        n_layers: int,
    ):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    d_ff,
                    n_head,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        dec: Tensor,
        enc_out: Tensor,
        self_attn_mask,
        dec_enc_attn_mask,
    ):
        for layer in self.layers:
            dec, self_attn, dec_enc_attn = layer(
                dec, enc_out, self_attn_mask, dec_enc_attn_mask
            )

        return dec, self_attn, dec_enc_attn
