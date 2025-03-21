from typing import List

import torch
import torch.utils.data as Data
from torch import LongTensor, Tensor

src_vocab = {
    "P": 0,
    "ich": 1,
    "mochte": 2,
    "ein": 3,
    "bier": 4,
    "cola": 5,
}
tgt_vocab = {
    "P": 0,
    "i": 1,
    "want": 2,
    "a": 3,
    "beer": 4,
    "coke": 5,
    "S": 6,
    "E": 7,
    ".": 8,
}
idx2word = {v: k for k, v in tgt_vocab.items()}


def make_data(
    data: List[List[str]],
) -> tuple[
    LongTensor,
    LongTensor,
    LongTensor,
]:
    enc_input = []
    dec_input = []
    dec_output = []
    for i in range(len(data)):
        enc_input.append([src_vocab[word] for word in data[i][0].split()])
        dec_input.append([tgt_vocab[word] for word in data[i][1].split()])
        dec_output.append([tgt_vocab[word] for word in data[i][2].split()])

    return (
        LongTensor(enc_input),
        LongTensor(dec_input),
        LongTensor(dec_output),
    )


class TransformerDataset(Data.Dataset):
    def __init__(
        self,
        enc_inputs: LongTensor,
        dec_inputs: LongTensor,
        dec_outputs: LongTensor,
    ) -> None:
        super().__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self) -> int:
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


sentences = [
    ["ich mochte ein bier P", "S i want a beer .", "i want a beer . E"],
    ["ich mochte ein cola P", "S i want a coke .", "i want a coke . E"],
]

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
dataset = TransformerDataset(enc_inputs, dec_inputs, dec_outputs)
loader = Data.DataLoader(
    TransformerDataset(enc_inputs, dec_inputs, dec_outputs),
    batch_size=2,
    shuffle=True,
)


def main():
    pass


if __name__ == "__main__":
    main()
