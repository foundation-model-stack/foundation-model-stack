import math
import unittest
import torch

from fms.modules.embedding import WordEmbedding
from fms.modules.positions import RotaryEmbedding


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
            assert y_pad[insert] == 0, f"Output pad token {y_pad[i]} is non-zero"
            y_ = y_pad[:insert] + y_pad[insert + 1 :]
            for i in range(len(y)):
                assert (
                    y[i] == y_[i]
                ), f"Index {i} of nonpadded output {y[i]} does not match padded output {y_[i]} with pad token {pad_id}"


class RotaryEmbeddingTests(unittest.TestCase):
    def test_args(self):
        q = torch.ones(2, 1, 4, 16, dtype=torch.float)  # b h s e
        k = 2 * torch.ones(2, 1, 4, 16, dtype=torch.float)  # b h s e
        rotary_embeddings = RotaryEmbedding(16, max_seq_len=32)

        with self.assertRaises(AssertionError):
            qr, kr = rotary_embeddings.adjusted_qk(q.squeeze(), k)

        with self.assertRaises(AssertionError):
            qr, kr = rotary_embeddings.adjusted_qk(q, k.squeeze())

        # This should not throw, as position_ids is optional
        qr, kr = rotary_embeddings.adjusted_qk(q, k)

        # This should not throw
        qr, kr = rotary_embeddings.adjusted_qk(q, k, torch.arange(0, q.size(2), device=q.device, dtype=torch.long), None)

        with self.assertRaises(IndexError):
            qr, kr = rotary_embeddings.adjusted_qk(q, k, torch.arange(0, q.size(2), device=q.device, dtype=torch.float), None)

    def test_math(self):
        q = torch.tensor([[1, 0], [1, 0]], dtype=torch.float).unsqueeze(0).unsqueeze(0)  # b h s e
        k = 2 * torch.tensor([[1, 0], [1, 0]], dtype=torch.float).unsqueeze(0).unsqueeze(0)  # b h s e
        rotary_embeddings = RotaryEmbedding(2, ratio=1, max_seq_len=2)

        qr, kr = rotary_embeddings.adjusted_qk(q, k)

        rot0 = torch.tensor([[math.cos(0), -math.sin(0)], [math.sin(0), math.cos(0)]])
        rot1 = torch.tensor([[math.cos(1), -math.sin(1)], [math.sin(1), math.cos(1)]])

        torch.testing.assert_close(torch.matmul(rot0, q[..., 0, :].squeeze()), qr[..., 0, :].squeeze())
        torch.testing.assert_close(torch.matmul(rot1, q[..., 1, :].squeeze()), qr[..., 1, :].squeeze())
        torch.testing.assert_close(torch.matmul(rot0, k[..., 0, :].squeeze()), kr[..., 0, :].squeeze())
        torch.testing.assert_close(torch.matmul(rot1, k[..., 1, :].squeeze()), kr[..., 1, :].squeeze())

    def test_pair_math(self):
        q = torch.tensor([[0, 1, 2, 3], [0, -1, 2, -3]], dtype=torch.float).unsqueeze(0).unsqueeze(0)  # b h s e
        k = torch.tensor([[1, -1, 1, -1], [1, 1, 1, 1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)  # b h s e
        rotary_embeddings = RotaryEmbedding(4, max_seq_len=2)
        qr, kr = rotary_embeddings.adjusted_qk(q, k)
        orig_dotp = q @ k.transpose(2, 3)
        rotated_dotp = qr @ kr.transpose(2, 3)

        # If two pairs of k/q have the same dot product before rotation,
        # and the same amount of rotation is applied to both pairs,
        # they'd have the same dot product after rotation
        # (even for cases where the two pairs are different k and q).
        torch.testing.assert_close(rotated_dotp[0, 0, 1, 0], rotated_dotp[0, 0, 0, 1])

    def test_left_padding(self):
        q = torch.ones(2, 1, 4, 16, dtype=torch.float)  # b h s e
        k = 2 * torch.ones(2, 1, 4, 16, dtype=torch.float)  # b h s e
        rotary_embeddings = RotaryEmbedding(16, max_seq_len=32)

        qr, kr = rotary_embeddings.adjusted_qk(q, k)

        qr2, kr2 = rotary_embeddings.adjusted_qk(q, k, torch.tensor([[i for i in range(4)], [1] + [i for i in range(3)]]))

        torch.testing.assert_close(qr[0], qr2[0])
        torch.testing.assert_close(qr[0, :, 1], qr2[1, :, 0])

        torch.testing.assert_close(kr[0], kr2[0])
        torch.testing.assert_close(kr[0, :, 1], kr2[1, :, 0])

    def test_long_sequences(self):
        q = torch.ones(2, 1, 64, 16, dtype=torch.float)  # b h s e
        k = 2 * torch.ones(2, 1, 64, 16, dtype=torch.float)  # b h s e
        rotary_embeddings = RotaryEmbedding(16, max_seq_len=32)

        # This should not throw, as we're within length
        qr, kr = rotary_embeddings.adjusted_qk(q[:, :, 0:31, :], k[:, :, 0:31, :])

        # With this codebase we never hit the out of bounds error
        qr, kr = rotary_embeddings.adjusted_qk(q, k)

        # rotary_embeddings.compute_freqs_cis(64)

        # # This should not throw, as we're within length
        # qr, kr = rotary_embeddings(q, k)

    def test_invariant_dotp(self):
        q = torch.normal(0, 1, (4, 8, 100, 128))  # b h s e
        k = torch.normal(0, 1, (4, 8, 100, 128))  # b h s e
        rotary_embeddings = RotaryEmbedding(128, max_seq_len=256)

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

    def test_relativity(self):
        embedding = torch.nn.Embedding(3, 256)
        qw = torch.nn.Linear(256, 256)
        kw = torch.nn.Linear(256, 256)
        q = qw(embedding(torch.tensor([[0, 1, 2, 0, 1, 2]]))).view(1, 6, 8, 32).transpose(1, 2)  # b h s e
        k = kw(embedding(torch.tensor([[0, 1, 2, 0, 1, 2]]))).view(1, 6, 8, 32).transpose(1, 2)  # b h s e
        rotary_embeddings = RotaryEmbedding(32, max_seq_len=128)

        qr, kr = rotary_embeddings.adjusted_qk(q, k)
        rotated_dotp = qr @ kr.transpose(2, 3)

        # if we have something like [ the blue dog the blue dog ],
        # then we'd expect the dot product of k.q (with relative position information applied)
        # to be the same for "the" dot "dog" in both the first and second occurance
        # (since they have the same distance), but not for q0 dot k5
        # (since they're farther apart).
        torch.testing.assert_close(rotated_dotp[0, 0, 0, 2], rotated_dotp[0, 0, 3, 5])
        assert torch.abs(rotated_dotp[0, 0, 0, 2] - rotated_dotp[0, 0, 0, 5]) > 1e-5
