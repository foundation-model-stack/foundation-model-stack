import math
import unittest

import torch

from fms.modules.positions import RotaryEmbedding


class RotaryEmbeddingTests(unittest.TestCase):
    def test_args(self):
        q = torch.ones(2, 4, 1, 16, dtype=torch.float)  # b s h e
        k = 2 * torch.ones(2, 4, 1, 16, dtype=torch.float)  # b s h e
        rotary_embeddings = RotaryEmbedding(16, max_seq_len=32)

        with self.assertRaises(AssertionError):
            qr, kr = rotary_embeddings.adjusted_qk(q.squeeze(), k)

        with self.assertRaises(AssertionError):
            qr, kr = rotary_embeddings.adjusted_qk(q, k.squeeze())

        # This should not throw, as position_ids is optional
        qr, kr = rotary_embeddings.adjusted_qk(q, k)

        # This should not throw
        qr, kr = rotary_embeddings.adjusted_qk(
            q,
            k,
            torch.arange(0, q.size(1), device=q.device, dtype=torch.long).unsqueeze(0),
            None,
        )

        with self.assertRaises(IndexError):
            qr, kr = rotary_embeddings.adjusted_qk(
                q,
                k,
                torch.arange(
                    0, q.size(1), device=q.device, dtype=torch.float
                ).unsqueeze(0),
                None,
            )

    def test_math(self):
        q = (
            torch.tensor([[1, 0], [1, 0]], dtype=torch.float).unsqueeze(0).unsqueeze(2)
        )  # b s h e
        k = 2 * torch.tensor([[1, 0], [1, 0]], dtype=torch.float).unsqueeze(
            0
        ).unsqueeze(2)  # b s h e
        rotary_embeddings = RotaryEmbedding(2, ratio=1, max_seq_len=2)

        qr, kr = rotary_embeddings.adjusted_qk(q, k)

        rot0 = torch.tensor([[math.cos(0), -math.sin(0)], [math.sin(0), math.cos(0)]])
        rot1 = torch.tensor([[math.cos(1), -math.sin(1)], [math.sin(1), math.cos(1)]])

        torch.testing.assert_close(
            torch.matmul(rot0, q[:, 0].squeeze()), qr[:, 0].squeeze()
        )
        torch.testing.assert_close(
            torch.matmul(rot1, q[:, 1].squeeze()), qr[:, 1].squeeze()
        )
        torch.testing.assert_close(
            torch.matmul(rot0, k[:, 0].squeeze()), kr[:, 0].squeeze()
        )
        torch.testing.assert_close(
            torch.matmul(rot1, k[:, 1].squeeze()), kr[:, 1].squeeze()
        )

    def test_pair_math(self):
        q = (
            torch.tensor([[0, 1, 2, 3], [0, -1, 2, -3]], dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(2)
        )  # b s h e
        k = (
            torch.tensor([[1, -1, 1, -1], [1, 1, 1, 1]], dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(2)
        )  # b s h e
        rotary_embeddings = RotaryEmbedding(4, max_seq_len=2)
        qr, kr = rotary_embeddings.adjusted_qk(q, k)

        q = q.transpose(1, 2)  # b h s e
        k = k.transpose(1, 2)  # b h s e
        qr = qr.transpose(1, 2)  # b h s e
        kr = kr.transpose(1, 2)  # b h s e

        rotated_dotp = qr @ kr.transpose(2, 3)

        # If two pairs of k/q have the same dot product before rotation,
        # and the same amount of rotation is applied to both pairs,
        # they'd have the same dot product after rotation
        # (even for cases where the two pairs are different k and q).
        torch.testing.assert_close(rotated_dotp[0, 0, 1, 0], rotated_dotp[0, 0, 0, 1])

    def test_left_padding(self):
        q = torch.ones(2, 4, 1, 16, dtype=torch.float)  # b s h e
        k = 2 * torch.ones(2, 4, 1, 16, dtype=torch.float)  # b s h e
        rotary_embeddings = RotaryEmbedding(16, max_seq_len=32)

        # First test that left-padding works as expected, with all the rotations moved right by one
        qr, kr = rotary_embeddings.adjusted_qk(q, k)

        qr2, kr2 = rotary_embeddings.adjusted_qk(
            q, k, torch.tensor([[i for i in range(4)], [1] + [i for i in range(3)]])
        )

        torch.testing.assert_close(qr[0], qr2[0])
        torch.testing.assert_close(qr[0, 1], qr2[1, 0])

        torch.testing.assert_close(kr[0], kr2[0])
        torch.testing.assert_close(kr[0, 1], kr2[1, 0])

        # Then test the need for position_ids in the API to ensure semantic correctnes
        q = torch.normal(0, 1, (2, 8, 1, 16))  # b s h e
        k = torch.normal(0, 1, (2, 8, 1, 16))  # b s h e

        # First generate a qr, kr that will act as kv-cache and the correct answers given one padding token on second row
        qr_cache, kr_cache = rotary_embeddings.adjusted_qk(
            q[:, -1:, :, :],
            k[:, -1:, :, :],
            position_ids=torch.tensor([list(range(7)), [1] + list(range(6))]),
        )
        qr_correct, kr_correct = rotary_embeddings.adjusted_qk(
            q, k, torch.tensor([list(range(8)), [1] + list(range(7))])
        )

        # Prove that without position_ids the cached position information is lost and results are incorrect
        qr_bad, kr_bad = rotary_embeddings.adjusted_qk(
            q[:, -1:, :, :],
            k[:, -1:, :, :],
            past_kv_state=(qr_cache, kr_cache),
            use_cache=True,
        )
        qr_good, kr_good = rotary_embeddings.adjusted_qk(
            q[:, -1:, :, :],
            k[:, -1:, :, :],
            past_kv_state=(qr_cache, kr_cache),
            use_cache=True,
            position_ids=torch.tensor([[7], [6]]),
        )

        torch.testing.assert_close(qr_good, qr_correct[:, -1:, :, :])
        torch.testing.assert_close(kr_good, kr_correct[:, -1:, :, :])

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(qr_bad, qr_correct[:, -1:, :, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(kr_bad, kr_correct[:, -1:, :, :])

    def test_long_sequences(self):
        q = torch.ones(2, 64, 1, 16, dtype=torch.float)  # b s h e
        k = 2 * torch.ones(2, 64, 1, 16, dtype=torch.float)  # b s h e
        rotary_embeddings = RotaryEmbedding(16, max_seq_len=32)

        # This should not throw, as we're within length
        qr, kr = rotary_embeddings.adjusted_qk(q[:, 0:31, :, :], k[:, 0:31, :, :])

        # With this codebase we should hit an out-of-bounds error
        # Without ntk-scaling, the max_seq_len is fixed and asking
        # for more should give an error
        with self.assertRaises(IndexError):
            qr, kr = rotary_embeddings.adjusted_qk(q, k)

    def test_invariant_dotp(self):
        q = torch.normal(0, 1, (4, 100, 8, 128))  # b s h e
        k = torch.normal(0, 1, (4, 100, 8, 128))  # b s h e
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
        q = qw(embedding(torch.tensor([[0, 1, 2, 0, 1, 2]]))).view(
            1, 6, 8, 32
        )  # b s h e
        k = kw(embedding(torch.tensor([[0, 1, 2, 0, 1, 2]]))).view(
            1, 6, 8, 32
        )  # b s h e
        rotary_embeddings = RotaryEmbedding(32, max_seq_len=128)

        qr, kr = rotary_embeddings.adjusted_qk(q, k)
        qr = qr.transpose(1, 2)  # b h s e
        kr = kr.transpose(1, 2)  # b h s e
        rotated_dotp = qr @ kr.transpose(2, 3)

        # if we have something like [ the blue dog the blue dog ],
        # then we'd expect the dot product of k.q (with relative position information applied)
        # to be the same for "the" dot "dog" in both the first and second occurance
        # (since they have the same distance), but not for q0 dot k5
        # (since they're farther apart).
        torch.testing.assert_close(rotated_dotp[0, 0, 0, 2], rotated_dotp[0, 0, 3, 5])
        assert torch.abs(rotated_dotp[0, 0, 0, 2] - rotated_dotp[0, 0, 0, 5]) > 1e-5

    def test_ntk(self):
        # B x S x H x Eh
        B = 3
        H = 5
        S = 10
        DIM = 50
        q = torch.randn((B, S, H, DIM))
        k = torch.randn((B, S, H, DIM))

        e = RotaryEmbedding(DIM, max_seq_len=S, ntk_scaling=False)
        adj_q, adj_k = e.adjusted_qk(q, k)
        ntk = RotaryEmbedding(DIM, max_seq_len=S, ntk_scaling=True)
        ntk_q, ntk_k = ntk.adjusted_qk(q, k)

        # <= max_seq_len, results should be the same with ntk_scaling.
        torch.testing.assert_close(adj_q, ntk_q)
        torch.testing.assert_close(adj_k, ntk_k)

        scaled_ratio = 10_000 / 2 ** (DIM / (DIM - 2))
        ntk = RotaryEmbedding(
            DIM, max_seq_len=S / 2, ratio=scaled_ratio, ntk_scaling=True
        )
        ntk_q, ntk_k = ntk.adjusted_qk(q, k)
        # being double the length is equivalent to being (approximately) half the base
        torch.testing.assert_close(adj_q, ntk_q)
        torch.testing.assert_close(adj_k, ntk_k)
