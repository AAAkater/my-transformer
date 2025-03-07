from torch import Tensor, nn


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden: int,
    ) -> None:
        super(PositionWiseFeedForward, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
