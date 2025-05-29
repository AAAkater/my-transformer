import math

from torch import Tensor, nn


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        d_k = q.size(-1)

        # Q * K^T/sqrt(d_k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        # 得出的scores是每个维度(d_1-d_v)都考虑了在当前维度(这一列)
        # 当前token对所有token的注意力后更新的新的值，
        # 换言之每个维度d是相互独立的，
        # 每个维度考虑自己的所有token的注意力，
        # 所以可以理解成1列扩展到多列

        # 加入mask矩阵
        scores += mask
        attn: Tensor = self.softmax(scores)

        # 返回的attn: [batch_size, n_heads, seq_len, d_k]本质上还是batch_size个句子，
        # 只不过每个句子中词向量维度512被分成了8个部分，分别由8个头各自看一部分，
        # 每个头算的是整个句子(一列)的512/8=64个维度，最后按列拼接起来
        return attn @ v
