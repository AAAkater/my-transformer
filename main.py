from torch import Tensor, nn
import torch


class LayerNorm(nn.Module):

    def __init__(self, feature: int, eps: float = 1e-6):
        """

        Args:
            feature (int): self-attention的x的大小
            eps (float, optional): _description_. Defaults to 1e-6.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x: Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SubLayoutConnection(nn.Module):
    """
    残差+layer_norm

    Args:
        nn (_type_): _description_
    """

    def __init__(self, size: int, dropout: float = 0.1):
        super(
            SubLayoutConnection,
            self,
        ).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sub_layer):
        """

        Args:
            x (Tensor): self-attention的输入
            sub_layer (_type_): self-attention

        Returns:
            _type_: _description_
        """
        return self.dropout(self.layer_norm(x + sub_layer(x)))
