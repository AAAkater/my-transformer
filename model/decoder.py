from torch import Tensor, nn

from model.block.my_decoder_layer import DecoderLayer
from model.embedding.my_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size: int,
        max_len: int,
        d_model: int,
        d_ff: int,
        n_head: int,
        n_layers: int,
    ):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(
            dec_voc_size,
            d_model,
            max_len,
        )
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
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(
        self,
        dec: Tensor,
        enc_out: Tensor,
        trg_mask: Tensor,
        src_mask: Tensor,
    ):
        trg: Tensor = self.emb(dec)
        for layer in self.layers:
            dec = layer(trg, enc_out, trg_mask, src_mask)
        out_put: Tensor = self.linear(trg)
        return out_put
