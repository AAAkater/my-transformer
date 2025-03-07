from torch import Tensor, nn

from model.embedding.my_positional_encoding import PositionalEncoding
from model.embedding.my_token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
    ) -> None:
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)

    def forward(self, x: Tensor) -> Tensor:
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return tok_emb + pos_emb
