import itertools
from collections import Counter

import spacy
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer

spacy_zh = spacy.load("zh_core_web_sm")
spacy_en = spacy.load("en_core_web_sm")


class WordDataset(Dataset):
    TOTAL_DATA_NUM = 5000

    def __init__(self, src_file: str, tgt_file: str):
        self.src_lines = open(src_file, encoding="utf-8").readlines()
        self.tgt_lines = open(tgt_file, encoding="utf-8").readlines()

        self.src_lines = self.src_lines[: self.TOTAL_DATA_NUM]
        self.tgt_lines = self.tgt_lines[: self.TOTAL_DATA_NUM]

        self.tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")
        self.tokenizer_zh = get_tokenizer("spacy", language="zh_core_web_sm")

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, index: int):
        src_line = self.src_lines[index].strip()
        tgt_line = self.tgt_lines[index].strip()

        src_tokens = [tok.text for tok in spacy_zh(src_line)]
        tgt_tokens = [tok.text for tok in spacy_en(tgt_line)]

        return src_tokens, tgt_tokens


class NumberDataset(Dataset):
    TOTAL_DATA_NUM = 5000

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_vocab: dict,
        tgt_vocab: dict,
        max_seq_len: int,
    ):
        self.src_lines = open(src_file, encoding="utf-8").readlines()

        self.tgt_lines = open(tgt_file, encoding="utf-8").readlines()
        self.src_lines = self.src_lines[: self.TOTAL_DATA_NUM]
        self.tgt_lines = self.tgt_lines[: self.TOTAL_DATA_NUM]

        self.tokenizer_en = get_tokenizer("spacy", language="en_core_web_sm")
        self.tokenizer_zh = get_tokenizer("spacy", language="zh_core_web_sm")

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index: int):
        # 获取原始句子
        src_line = self.src_lines[index].strip()
        tgt_line = self.tgt_lines[index].strip()

        # 分词
        src_tokens = [tok.text for tok in spacy_zh(src_line)]
        trg_tokens = [tok.text for tok in spacy_en(tgt_line)]

        # 转换为索引
        src_indices = [
            self.src_vocab.get(
                token,
                3,
            )
            for token in src_tokens
        ]
        tgt_indices = [
            self.tgt_vocab.get(
                token,
                3,
            )
            for token in trg_tokens
        ]

        # 添加特殊token
        src_indices = [1] + src_indices + [2]  # <sos>和<eos>
        tgt_indices = [1] + tgt_indices + [2]

        # 填充到max_len
        src_padded = src_indices[: self.max_seq_len] + [0] * (
            self.max_seq_len - len(src_indices)
        )
        tgt_padded = tgt_indices[: self.max_seq_len] + [0] * (
            self.max_seq_len - len(tgt_indices)
        )

        return torch.tensor(src_padded, dtype=torch.long), torch.tensor(
            tgt_padded, dtype=torch.long
        )


all_vocab = {
    "<pad>": 0,
    "<sos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}
zh_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
en_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}


def main():
    MIN_FREQ = 2

    zh_vocab_file = "./data/words/zh_vocab.pkl"
    en_vocab_file = "./data/words/en_vocab.pkl"
    src_file = "./data/words/chinese.txt"
    tgt_file = "./data/words/english.txt"

    dataset = WordDataset(src_file, tgt_file)

    zh_tokens = list(
        itertools.chain.from_iterable(
            src_tokens for src_tokens, tgt_tokens in dataset
        )
    )
    en_tokens = list(
        itertools.chain.from_iterable(
            tgt_tokens for src_tokens, tgt_tokens in dataset
        )
    )

    zh_vocab_counter = Counter(zh_tokens)
    en_vocab_counter = Counter(en_tokens)

    zh_total = len(zh_vocab)

    for index, (token, freq) in enumerate(zh_vocab_counter.items()):
        if freq >= MIN_FREQ:
            zh_vocab.update({token: zh_total})
            zh_total += 1

    en_total = len(en_vocab)
    for index, (token, freq) in enumerate(en_vocab_counter.items()):
        if freq >= MIN_FREQ:
            en_vocab.update({token: en_total})
            en_total += 1

    with open(zh_vocab_file, "wb") as zh_vocab_file:
        torch.save(zh_vocab, zh_vocab_file)

    with open(en_vocab_file, "wb") as en_vocab_file:
        torch.save(en_vocab, en_vocab_file)


if __name__ == "__main__":
    main()
