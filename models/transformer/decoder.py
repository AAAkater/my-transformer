from torch import Tensor, nn

from models.transformer.block.my_decoder_layer import DecoderLayer
from models.transformer.embedding.my_embedding import TransformerEmbedding


class Decoder(nn.Module):
    def __init__(
        self,
        dec_vocab_size: int,
        max_seq_len: int,
        d_model: int,
        d_ff: int,
        n_head: int,
        n_layers: int,
    ):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(
            dec_vocab_size,
            d_model,
            max_seq_len,
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
        self.linear = nn.Linear(d_model, dec_vocab_size)

    def forward(
        self,
        dec_out: Tensor,
        enc_out: Tensor,
        tgt_mask: Tensor,
        src_mask: Tensor,
    ):
        tgt: Tensor = self.emb(dec_out)
        for layer in self.layers:
            tgt = layer(tgt, enc_out, tgt_mask, src_mask)
        out_put: Tensor = self.linear(tgt)
        return out_put
