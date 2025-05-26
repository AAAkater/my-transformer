from torch import Tensor, nn


class Bert(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_ff: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden = d_ff
        self.n_layers = n_layers
        self.n_heads = n_heads

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = nn.Embedding(vocab_size, d_ff)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_ff,
                    nhead=n_heads,
                    dim_feedforward=d_ff * 4,
                    dropout=dropout_rate,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: Tensor):
        x = self.embedding(x)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        return x
