import sentencepiece as spm


def train_tokenizer(
    zh_file: str,
    en_file: str,
    vocab_size: int,
):
    spm.SentencePieceTrainer.Train(
        input=zh_file,
        model_prefix="spm_zh",
        vocab_size=vocab_size,
        character_coverage=0.9995,  # 中文需要较低的覆盖率
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<sep>", "<cls>", "<mask>"],
        split_digits=True,  # 将数字分开
        remove_extra_whitespaces=True,
    )

    spm.SentencePieceTrainer.Train(
        input=en_file,
        model_prefix="spm_en",
        vocab_size=vocab_size,
        character_coverage=1.0,  # 英文需要较高的覆盖率
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=["<sep>", "<cls>", "<mask>"],
        split_digits=True,  # 将数字分开
        remove_extra_whitespaces=True,
        normalization_rule_name="nmt_nfkc",  # 英文使用NFKC标准化
    )


class SentencePieceVocab:
    def __init__(self, sp_model_path: str):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(sp_model_path)
        # 构建id到token的映射
        self.id2token = {
            i: self.sp_model.IdToPiece(i)
            for i in range(self.sp_model.GetPieceSize())
        }

        # 构建token到id的映射
        self.token2id = {v: k for k, v in self.id2token.items()}

    def encode_as_ids(self, text: str) -> list[int]:
        return self.sp_model.EncodeAsIds(text)

    def encode_as_pieces(self, text: str) -> list[str]:
        return self.sp_model.EncodeAsPieces(text)

    def decode(self, ids: list[int]) -> str:
        return self.sp_model.DecodeIds(ids)

    def __len__(self):
        return len(self.id2token)

    def get_vocab_size(self) -> int:
        return self.sp_model.GetPieceSize()

    def bos_id(self) -> int:
        return self.sp_model.bos_id()

    def eos_id(self) -> int:
        return self.sp_model.eos_id()

    def pad_id(self) -> int:
        return self.sp_model.pad_id()


def main():
    # Example usage
    zh_path = "data/words/chinese.txt"
    en_path = "data/words/english.txt"
    train_tokenizer(
        zh_file=zh_path,
        en_file=en_path,
        vocab_size=32000,
    )
    print("tokenizer trained successfully.")


if __name__ == "__main__":
    # main()

    # 创建词汇表实例
    zh_vocab = SentencePieceVocab("./data/words/spm_zh.model")
    en_vocab = SentencePieceVocab("./data/words/spm_en.model")

    print(f"中文词汇表大小: {zh_vocab.get_vocab_size()}")
    print(f"英文词汇表大小: {en_vocab.get_vocab_size()}")
