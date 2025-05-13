import torch
from torch import Tensor, nn

from model.decoder import Decoder
from model.encoder import Encoder


class Transformer(nn.Module):
    """
    transformer
    """

    def __init__(
        self,
        src_pad_idx: int,
        trg_pad_idx: int,
        trg_sos_idx: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_head: int = 8,
        n_layer: int = 6,
        d_ff: int = 2048,
    ):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.encoder = Decoder(
            dec_voc_size=src_vocab_size,
            max_len=5000,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            n_layers=n_layer,
        )

        self.decoder = Encoder(
            enc_voc_size=src_vocab_size,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layer,
            max_len=5000,
            ffn_hidden=d_ff,
        )

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
    ) -> Tensor:
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src: Tensor = self.encoder(src, src_mask)

        output: Tensor = self.decoder(
            trg,
            enc_src,
            trg_mask,
            src_mask,
        )

        return output

    def make_src_mask(self, src: Tensor):
        src_mask = (
            (src != self.src_pad_idx)
            .unsqueeze(1)
            .unsqueeze(
                2,
            )
        )
        return src_mask

    def make_trg_mask(self, trg: Tensor):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(
            torch.ones(trg_len, trg_len),
        ).to(dtype=torch.uint8)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask


if __name__ == "__main__":
    pass
