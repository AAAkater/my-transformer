import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.decoder import Decoder
from model.encoder import Encoder


class Transformer(nn.Module):
    """
    transformer
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        head_nums: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout_rate: float = 0.1,
    ):
        super(Transformer, self).__init__()
        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    head_nums,
                    dim_feedforward,
                    dropout_rate,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    head_nums,
                    dim_feedforward,
                    dropout_rate,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor = None,
        tgt_mask: Tensor = None,
    ) -> Tensor:
        """
        向前传播

        Args:
            src (Tensor): 源序列张量
            tgt (Tensor): 目标序列张量
            src_mask (Tensor, optional): 源序列张量掩码. Defaults to None.
            tgt_mask (Tensor, optional): 目标序列张量掩码. Defaults to None.

        Returns:
            Tensor: _description_
        """
        for encoder in self.encoder:
            src = encoder(src, src_mask)

        for decoder in self.decoder:
            tgt = decoder(tgt, src, tgt_mask, src_mask)

        return F.log_softmax(self.generator(tgt), dim=-1)


if __name__ == "__main__":
    src = torch.randint(0, 100, (10, 32))
    tgt = torch.randint(0, 100, (20, 32))
    model = Transformer(100, 100)
    out: Tensor = model(src, tgt)
    print(out.shape)
