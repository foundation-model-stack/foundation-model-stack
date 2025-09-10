import torch


def apply_abs_pos_emb(
    emb: torch.nn.Embedding,
    pos_emb: torch.nn.Embedding,
    inp: torch.Tensor,
    padding_idx: int = 0,
):
    out = emb(inp)
    pos = torch.arange(inp.size(1)).unsqueeze(0)
    is_pad = inp == padding_idx
    pos = pos.sub(is_pad.cumsum(1))
    # In case of left-padding, prevent negative indices (get zeroed anyways)
    pos = pos.clamp(min=0)
    return out.addcmul(pos_emb(pos), ~is_pad.unsqueeze(-1))


def test_abs_pos_padding():
    L = 50
    for pad_id in range(L):
        for insert in range(L):
            m = torch.nn.Embedding(L, 1, padding_idx=pad_id)
            p = torch.nn.Embedding(L, 1)
            x = list(range(L))
            x = x[:pad_id] + x[pad_id + 1 :]
            x_tensor = torch.tensor(x, dtype=torch.int64).unsqueeze(0)
            x_pad = x[:insert] + [pad_id] + x[insert:]
            x_pad_tensor = torch.tensor(x_pad, dtype=torch.int64).unsqueeze(0)
            y = apply_abs_pos_emb(m, p, x_tensor, pad_id).flatten().tolist()
            y_pad = apply_abs_pos_emb(m, p, x_pad_tensor, pad_id).flatten().tolist()
            assert y_pad[insert] == 0, f"Output pad token {y_pad[insert]} is non-zero"
            y_ = y_pad[:insert] + y_pad[insert + 1 :]
            for i in range(len(y)):
                assert y[i] == y_[i], (
                    f"Index {i} of nonpadded output {y[i]} does not match padded output {y_[i]} with pad token {pad_id}"
                )
