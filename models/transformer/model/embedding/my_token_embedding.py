from torch import nn


class TokenEmbedding(nn.Embedding):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
    ) -> None:
        super(TokenEmbedding, self).__init__(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )
