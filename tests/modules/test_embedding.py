import torch

from fms.modules.embedding import WordEmbedding


def test_abs_pos_padding():
    L = 50
    for pad_id in range(L):
        for insert in range(L):
            m = WordEmbedding(
                L,
                1,
                padding_idx=pad_id,
                max_pos=L,
                abs_pos=True,
                reversible=False,
                tie_weights=False,
            )
            x = list(range(L))
            x = x[:pad_id] + x[pad_id + 1 :]
            x_pad = x[:insert] + [pad_id] + x[insert:]
            y = m(torch.IntTensor(x).unsqueeze(0)).flatten().tolist()
            y_pad = m(torch.IntTensor(x_pad).unsqueeze(0)).flatten().tolist()
            assert y_pad[insert] == 0, f"Output pad token {y_pad[insert]} is non-zero"
            y_ = y_pad[:insert] + y_pad[insert + 1 :]
            for i in range(len(y)):
                assert y[i] == y_[i], (
                    f"Index {i} of nonpadded output {y[i]} does not match padded output {y_[i]} with pad token {pad_id}"
                )
