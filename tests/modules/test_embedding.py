import math

import torch

from fms.modules.embedding import WordEmbedding, AbsolutePositionEmbedding


def test_absolute_position_embedding_correct_pads():
    """Test that absolute position embeddings correctly corrects for pads in middle of sequence"""
    L = 50
    for pad_id in range(L):
        for insert in range(L):
            m = AbsolutePositionEmbedding(L, 1, padding_idx=pad_id, max_pos=L)
            x = list(range(L))
            x = x[:pad_id] + x[pad_id + 1 :]
            x_pad = x[:insert] + [pad_id] + x[insert:]
            y = m(torch.IntTensor(x).unsqueeze(0)).flatten().tolist()
            y_pad = (
                m(torch.IntTensor(x_pad).unsqueeze(0), correct_pads=True)
                .flatten()
                .tolist()
            )
            assert y_pad[insert] == 0, f"Output pad token {y_pad[i]} is non-zero"
            y_ = y_pad[:insert] + y_pad[insert + 1 :]
            for i in range(len(y)):
                assert (
                    y[i] == y_[i]
                ), f"Index {i} of nonpadded output {y[i]} does not match padded output {y_[i]} with pad token {pad_id}"


def test_absolute_position_embedding_provides_position_ids():
    """test that absolution positional embeddings will use the given position ids properly when not correcting for pads and position_ids provided"""
    vocab_size = 50
    emb_dim = 1
    padding_idx = 0
    max_pos = 5
    emb = AbsolutePositionEmbedding(
        vocab_size, emb_dim, padding_idx=padding_idx, max_pos=max_pos
    )
    x = torch.IntTensor([0, 0, 0, 1, 2]).unsqueeze(0)
    position_ids = torch.IntTensor([1, 1, 1, 0, 1]).unsqueeze(0)
    actual = emb.forward(x, position_ids=position_ids)
    expected = (
        torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                emb.emb.weight[1].item() + emb.pos_emb.weight[0].item(),
                emb.emb.weight[2].item() + emb.pos_emb.weight[1].item(),
            ]
        )
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    assert torch.equal(expected, actual)


def test_absolute_position_embedding_provides_position_ids_correct_pads():
    """test that absolution positional embeddings will use the given position ids properly when correcting for pads and position_ids provided"""
    vocab_size = 50
    emb_dim = 1
    padding_idx = 0
    max_pos = 10
    emb = AbsolutePositionEmbedding(
        vocab_size, emb_dim, padding_idx=padding_idx, max_pos=max_pos
    )
    x = torch.IntTensor([0, 0, 0, 1, 2]).unsqueeze(0)
    position_ids = torch.IntTensor([5, 6, 7, 8, 9]).unsqueeze(0)
    actual = emb.forward(x, position_ids=position_ids, correct_pads=True)
    expected = (
        torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                emb.emb.weight[1].item() + emb.pos_emb.weight[5].item(),
                emb.emb.weight[2].item() + emb.pos_emb.weight[6].item(),
            ]
        )
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    assert torch.equal(expected, actual)
