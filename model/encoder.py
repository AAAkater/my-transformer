from torch import Tensor, nn

from model.block.my_encoder_layer import EncoderLayer
from model.embedding.my_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(
        self,
        enc_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        device,
    ):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
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

    def forward(self, x: Tensor, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
