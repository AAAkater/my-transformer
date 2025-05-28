from torch import Tensor, nn

from models.transformer.layers.my_scale_dot_product_attention import (
    ScaleDotProductAttention,
)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % 2 == 0, f"{d_model} must be even!"
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: Tensor,  # (batch_size, seq_len, d_model)
        k: Tensor,  # (batch_size, seq_len, d_model)
        v: Tensor,  # (batch_size, seq_len, d_model)
        mask: Tensor,
    ):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # Split into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size, n_head, seq_len, d_k)
        # where d_k = d_model // n_head
        q, k, v = self.split(q), self.split(k), self.split(v)

        out: Tensor = self.attention(q, k, v, mask)

        # Concatenate heads
        # (batch_size, n_head, seq_len, d_k) -> (batch_size, seq_len, d_model)
        out = self.concat(out)

        out: Tensor = self.linear(out)

        return out

    def split(self, x: Tensor) -> Tensor:
        # (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()

        d_k = d_model // self.n_head

        x = x.view(batch_size, seq_len, self.n_head, d_k).transpose(1, 2)
        # (batch_size, n_head, seq_len, d_k)
        return x

    def concat(self, x: Tensor) -> Tensor:
        batch_size, n_head, seq_len, d_k = x.size()

        x = (
            # (batch_size, seq_len, n_head, d_k)
            x.transpose(1, 2)
            # 保证内存连续
            .contiguous()
            # (batch_size, seq_len, d_model)
            .view(batch_size, seq_len, n_head * d_k)
        )

        return x
