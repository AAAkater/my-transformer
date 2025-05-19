import torch
from torch.utils.data import Dataset

from data.build_vocab import SentencePieceVocab


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_spm_model: SentencePieceVocab,
        tgt_spm_model: SentencePieceVocab,
        max_length: int = 50,
    ):
        self.src_spm = src_spm_model
        self.tgt_spm = tgt_spm_model

        # 读取文件
        with open(src_file, "r", encoding="utf-8") as f:
            self.src_sentences = [line.strip() for line in f]

        with open(tgt_file, "r", encoding="utf-8") as f:
            self.tgt_sentences = [line.strip() for line in f]

        assert len(self.src_sentences) == len(self.tgt_sentences), (
            "源语言和目标语言句子数量不匹配"
        )

        self.max_length = max_length

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx: int):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]

        # 使用SentencePiece编码
        src_ids = (
            [self.src_spm.bos_id()]
            + self.src_spm.encode_as_ids(src_sentence)
            + [self.src_spm.eos_id()]
        )
        tgt_ids = (
            [self.tgt_spm.bos_id()]
            + self.tgt_spm.encode_as_ids(tgt_sentence)
            + [self.tgt_spm.eos_id()]
        )

        # 截断或填充到固定长度
        src_ids = self._pad_or_truncate(src_ids, self.src_spm.pad_id())
        tgt_ids = self._pad_or_truncate(tgt_ids, self.tgt_spm.pad_id())

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }

    def _pad_or_truncate(self, token_ids: list[int], pad_id: int):
        # 截断
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length - 1] + [
                token_ids[-1]
            ]  # 保留EOS

        # 填充
        padding_length = self.max_length - len(token_ids)
        token_ids = token_ids + [pad_id] * padding_length

        return token_ids


zh_model = SentencePieceVocab("./data/words/spm_zh.model")
en_model = SentencePieceVocab("./data/words/spm_en.model")

# 创建数据集实例
dataset = TranslationDataset(
    src_file="data/words/english.txt",
    tgt_file="data/words/chinese.txt",
    src_spm_model=zh_model,
    tgt_spm_model=en_model,
    max_length=50,
)


if __name__ == "__main__":
    # 示例：查看分词效果
    sample_zh = "今天天气真好"
    sample_en = "The weather is nice today"

    print("中文分词示例:")
    print(f"原始句子: {sample_zh}")
    print(f"分词结果: {zh_model.encode_as_pieces(sample_zh)}")
    print(f"ID序列: {zh_model.encode_as_ids(sample_zh)}")

    print("\n英文分词示例:")
    print(f"原始句子: {sample_en}")
    print(f"分词结果: {en_model.encode_as_pieces(sample_en)}")
    print(f"ID序列: {en_model.encode_as_ids(sample_en)}")

    # 使用DataLoader加载数据
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        print(f"源语言批次形状: {batch['src'].shape}")
        print(f"目标语言批次形状: {batch['tgt'].shape}")
        break
