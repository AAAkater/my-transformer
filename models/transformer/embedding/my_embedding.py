from torch import Tensor, nn

from models.transformer.embedding.my_positional_encoding import (
    PositionalEncoding,
)
from models.transformer.embedding.my_token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        drop_rate: float = 0.1,
    ) -> None:
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_seq_len)
        self.drop_out = nn.Dropout(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        tok_emb: Tensor = self.tok_emb(x)
        pos_emb: Tensor = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
