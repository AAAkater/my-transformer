import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    batch_size: int = 128
    max_len: int = 256
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    ffn_hidden: int = 2048


settings = Settings()
