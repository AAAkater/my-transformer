import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ) -> None:
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0, f"{d_model} must be even!"

        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(
            0,
            max_len,
            dtype=torch.float,
        ).unsqueeze(dim=1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                step=2,
                dtype=torch.float,
            )
            * -(torch.log(Tensor(10000.0)) / d_model)
        )

        self.encoding[:, 0::2] = torch.sin(pos / div_term)
        self.encoding[:, 1::2] = torch.cos(pos / div_term)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]
