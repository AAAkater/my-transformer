import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    # 批次大小
    batch_size: int = 128
    # 一个句子最多包含的token数
    max_seq_len: int = 5000
    # 用来表示一个词的向量长度
    d_model: int = 512
    # Encoder Layer 和 Decoder Layer的个数
    n_layers: int = 6
    # 多头注意力中head的个数
    n_heads: int = 8
    # FFN的隐藏层神经元个数
    ffn_hidden: int = 2048
    # 分头后的q、k、v词向量长度
    d_k: int = 64
    drop_rate: float = 0.1


settings = Settings()


if __name__ == "__main__":
    print(settings.d_model)
