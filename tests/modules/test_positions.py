import math

import torch

from fms.modules.positions import RotaryEmbedding


def test_rotary_embeddings_math():
    q = (
        torch.tensor([[1, 0], [1, 0]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
    )  # b h s e
    k = 2 * torch.tensor([[1, 0], [1, 0]], dtype=torch.float).unsqueeze(0).unsqueeze(
        0
    )  # b h s e
    rotary_embeddings = RotaryEmbedding(2, 2)

    qr, kr = rotary_embeddings.adjusted_qk(q, k)

    rot0 = torch.tensor([[math.cos(0), -math.sin(0)], [math.sin(0), math.cos(0)]])
    rot1 = torch.tensor([[math.cos(1), -math.sin(1)], [math.sin(1), math.cos(1)]])

    torch.testing.assert_close(
        torch.matmul(rot0, q[..., 0, :].squeeze()), qr[..., 0, :].squeeze()
    )
    torch.testing.assert_close(
        torch.matmul(rot1, q[..., 1, :].squeeze()), qr[..., 1, :].squeeze()
    )
    torch.testing.assert_close(
        torch.matmul(rot0, k[..., 0, :].squeeze()), kr[..., 0, :].squeeze()
    )
    torch.testing.assert_close(
        torch.matmul(rot1, k[..., 1, :].squeeze()), kr[..., 1, :].squeeze()
    )


def test_rotary_embeddings_pair_math():
    q = (
        torch.tensor([[0, 1, 2, 3], [0, -1, 2, -3]], dtype=torch.float)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # b h s e
    k = (
        torch.tensor([[1, -1, 1, -1], [1, 1, 1, 1]], dtype=torch.float)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # b h s e
    rotary_embeddings = RotaryEmbedding(4, 2)

    qr, kr = rotary_embeddings.adjusted_qk(q, k)
    orig_dotp = q @ k.transpose(2, 3)
    rotated_dotp = qr @ kr.transpose(2, 3)

    print(orig_dotp, rotated_dotp)

    # If two pairs of k/q have the same dot product before rotation,
    # and the same amount of rotation is applied to both pairs,
    # they'd have the same dot product after rotation
    # (even for cases where the two pairs are different k and q).
    torch.testing.assert_close(rotated_dotp[0, 0, 1, 0], rotated_dotp[0, 0, 0, 1])


def test_rotary_embeddings_left_padding():
    q = torch.ones(2, 1, 4, 16, dtype=torch.float)  # b h s e
    k = 2 * torch.ones(2, 1, 4, 16, dtype=torch.float)  # b h s e
    rotary_embeddings = RotaryEmbedding(16, 32)

    qr, kr = rotary_embeddings.adjusted_qk(q, k, 0)
    # todo: fix and test calculation of start_pos
    # qr2, kr2 = rotary_embeddings(q, k, torch.tensor([0, 1]))

    # torch.testing.assert_close(qr[0], qr2[0])
    # torch.testing.assert_close(qr[0, :, 1], qr2[1, :, 0])

    # torch.testing.assert_close(kr[0], kr2[0])
    # torch.testing.assert_close(kr[0, :, 1], kr2[1, :, 0])


def test_rotary_embeddings_invariant_dotp():
    q = torch.normal(0, 1, (4, 8, 100, 128))  # b h s e
    k = torch.normal(0, 1, (4, 8, 100, 128))  # b h s e
    rotary_embeddings = RotaryEmbedding(128, 256)

    qr, kr = rotary_embeddings.adjusted_qk(q, k)

    orig_dotp = q @ k.transpose(2, 3)
    rotated_dotp = qr @ kr.transpose(2, 3)

    # Values that are rotated the same amount will have
    # no change in dot-product (i.e. a no-op when the
    # position of k and q are the same).
    # Tols are a little higher than usual due to some test flakiness
    torch.testing.assert_close(
        torch.diagonal(orig_dotp, dim1=2, dim2=3),
        torch.diagonal(rotated_dotp, dim1=2, dim2=3),
        atol=1e-4,
        rtol=1e-5,
    )


def test_rotary_embeddings_relativity():
    embedding = torch.nn.Embedding(3, 256)
    qw = torch.nn.Linear(256, 256)
    kw = torch.nn.Linear(256, 256)
    q = (
        qw(embedding(torch.tensor([[0, 1, 2, 0, 1, 2]])))
        .view(1, 6, 8, 32)
        .transpose(1, 2)
    )  # b h s e
    k = (
        kw(embedding(torch.tensor([[0, 1, 2, 0, 1, 2]])))
        .view(1, 6, 8, 32)
        .transpose(1, 2)
    )  # b h s e
    rotary_embeddings = RotaryEmbedding(32, 128)

    qr, kr = rotary_embeddings.adjusted_qk(q, k)
    rotated_dotp = qr @ kr.transpose(2, 3)

    # if we have something like [ the blue dog the blue dog ],
    # then we'd expect the dot product of k.q (with relative position information applied)
    # to be the same for "the" dot "dog" in both the first and second occurance
    # (since they have the same distance), but not for q0 dot k5
    # (since they're farther apart).
    torch.testing.assert_close(rotated_dotp[0, 0, 0, 2], rotated_dotp[0, 0, 3, 5])
    assert torch.abs(rotated_dotp[0, 0, 0, 2] - rotated_dotp[0, 0, 0, 5]) > 1e-5
