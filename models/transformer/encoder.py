from torch import Tensor, nn

from models.transformer.block.my_encoder_layer import EncoderLayer
from models.transformer.embedding.my_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(
        self,
        enc_voc_size: int,
        max_seq_len: int,
        d_model: int,
        ffn_hidden: int,
        n_head: int,
        n_layers: int,
    ):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            vocab_size=enc_voc_size,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    d_ff=ffn_hidden,
                    n_head=n_head,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc: Tensor, src_mask: Tensor):
        enc = self.emb(enc)

        for layer in self.layers:
            enc = layer(enc, src_mask)

        return enc
