import torch
from torch import Tensor, nn

from models.transformer.decoder import Decoder
from models.transformer.encoder import Encoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx: int,
        tgt_pad_idx: int,
        tgt_sos_idx: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_head: int = 8,
        n_layer: int = 6,
        d_ff: int = 2048,
    ):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.decoder = Decoder(
            dec_vocab_size=tgt_vocab_size,
            max_len=5000,
            d_model=d_model,
            n_head=n_head,
            d_ff=d_ff,
            n_layers=n_layer,
        )

        self.encoder = Encoder(
            enc_voc_size=src_vocab_size,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layer,
            max_len=5000,
            ffn_hidden=d_ff,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
    ) -> Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_src: Tensor = self.encoder(src, src_mask)

        output: Tensor = self.decoder(
            tgt,
            enc_src,
            tgt_mask,
            src_mask,
        )

        return output

    def make_src_mask(self, src: Tensor) -> Tensor:
        src_pad_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_pad_mask

    def make_tgt_mask(self, tgt: Tensor):
        # 填充掩码
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.shape[1]
        # 下三角掩码
        tgt_sub_mask = torch.tril(
            torch.ones(tgt_len, tgt_len),
        ).to(dtype=torch.uint8)
        # 合并掩码
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask


if __name__ == "__main__":
    pass
