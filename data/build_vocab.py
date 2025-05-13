import sentencepiece as spm


def train_tokenizer(
    input_file: str, vocab_size: int, model_prefix: str, lang: str
):
    """
    Trains a SentencePiece tokenizer with language-specific configurations.

    Args:
        input_file (str): Path to the input text file for training the tokenizer.
        vocab_size (int): Desired size of the vocabulary.
        model_prefix (str): Prefix for the output model files.
        lang (str): Language code ('zh' for Chinese or 'en' for English) which determines
            specific training parameters.

    Note:
        - For Chinese (lang='zh'): Uses lower character coverage (0.9995) and basic BPE model.
        - For English (lang='en'): Uses full character coverage (1.0), BPE model with NFKC
          normalization.
        - Both languages use the same special tokens: <sep>, <cls>, <mask> and split digits.
        - Default token IDs: pad=0, unk=1, bos=2, eos=3.

    Raises:
        ValueError: If lang is not 'zh' or 'en'.
    """
    if lang == "zh":
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
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
    elif lang == "en":
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=model_prefix,
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


if __name__ == "__main__":
    # Example usage

    zh_path = "data/words/chinese.txt"
    train_tokenizer(
        input_file=zh_path,
        vocab_size=32000,
        model_prefix="data/spm_model",
        lang="zh",
    )
    en_path = "data/words/english.txt"
    print("Chinese tokenizer trained successfully.")
    train_tokenizer(
        input_file=en_path,
        vocab_size=32000,
        model_prefix="data/spm_model_en",
        lang="en",
    )
