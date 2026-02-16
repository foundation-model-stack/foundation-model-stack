import math
import unittest

import pytest
import torch

from fms.modules.positions import RotaryEmbedding, PixtralRotaryEmbedding


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

        e = RotaryEmbedding(DIM, max_seq_len=S, scaling={})
        adj_q, adj_k = e.adjusted_qk(q, k)
        ntk = RotaryEmbedding(DIM, max_seq_len=S, scaling={"rope_type": "ntk"})
        ntk_q, ntk_k = ntk.adjusted_qk(q, k)

        # <= max_seq_len, results should be the same with ntk_scaling.
        torch.testing.assert_close(adj_q, ntk_q)
        torch.testing.assert_close(adj_k, ntk_k)

        scaled_ratio = 10_000 / 2 ** (DIM / (DIM - 2))
        ntk = RotaryEmbedding(
            DIM, max_seq_len=S / 2, ratio=scaled_ratio, scaling={"rope_type": "ntk"}
        )
        ntk_q, ntk_k = ntk.adjusted_qk(q, k)
        # being double the length is equivalent to being (approximately) half the base
        torch.testing.assert_close(adj_q, ntk_q)
        torch.testing.assert_close(adj_k, ntk_k)


class PixtralRotaryEmbeddingTest(unittest.TestCase):
    def test_shapes(self):
        # Configuration
        dim = 16
        ratio = 10_000.0
        image_size = 16
        patch_size = 4
        batch_size = 2
        num_heads = 2

        rope = PixtralRotaryEmbedding(dim, ratio, image_size, patch_size)

        self.assertEqual(rope.dim, dim)
        self.assertEqual(rope.max_patches_per_side, image_size // patch_size)

        # Create dummy query and key tensors
        q = torch.randn(batch_size, patch_size, num_heads, dim)
        k = torch.randn(batch_size, patch_size, num_heads, dim)

        # Test with explicit position_ids
        height = patch_size
        width = patch_size
        # Create a sample grid; this is similar to get_positions_in_meshgrid for pixtral
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        patch_pod_ids = torch.stack(mesh, dim=-1).reshape(-1, 2)
        # Expand the batch dim
        position_ids = patch_pod_ids.unsqueeze(0).repeat(batch_size, 1, 1)

        q_rot, k_rot = rope.adjusted_qk(q, k, position_ids=position_ids)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_requires_positions(self):
        """ensure that pixtral rope requires position IDs for 2D encoding."""
        # Configuration
        dim = 16
        ratio = 10_000.0
        image_size = 16
        patch_size = 4
        batch_size = 2
        num_heads = 2

        rope = PixtralRotaryEmbedding(dim, ratio, image_size, patch_size)

        self.assertEqual(rope.dim, dim)
        self.assertEqual(rope.max_patches_per_side, image_size // patch_size)

        # Create dummy query and key tensors
        q = torch.randn(batch_size, patch_size, num_heads, dim)
        k = torch.randn(batch_size, patch_size, num_heads, dim)

        # Test with explicit position_ids; this is allowed by the
        # position encoder superclass, but not for pixtral since
        # we need to know the grid layout (i.e., n x m and m x n
        # will have different positional encodings when n and m are
        # different).
        with pytest.raises(ValueError):
            rope.adjusted_qk(q, k, position_ids=None)

    def test_rope_at_position_0(self):
        """position (0,0) should have no rotation"""
        # Configuration
        dim = 16
        ratio = 10_000.0
        image_size = 16
        patch_size = 4
        batch_size = 1
        num_heads = 2

        rope = PixtralRotaryEmbedding(dim, ratio, image_size, patch_size)

        # Create dummy query and key tensors
        q = torch.randn(batch_size, patch_size, num_heads, dim)
        k = torch.randn(batch_size, patch_size, num_heads, dim)

        # Create 2D position_ids with batch dimension: [batch_size, seq_len, 2]
        # Position (0,0) means both height and width coordinates are 0
        position_ids = torch.zeros(batch_size, patch_size, 2, dtype=torch.long)

        # Apply pixtral rotary embeddings
        q_rot, _ = rope.adjusted_qk(q, k, position_ids=position_ids)

        torch.testing.assert_close(
            q_rot[0, 0, 0, :],
            q[0, 0, 0, :],
            atol=1e-6,
            rtol=1e-6,
            msg="Position (0,0) should have no rotation",
        )

    def test_hf_fms_equivalence(self):
        """Test that FMS Pixtral RoPE matches HF Transformers implementation"""
        try:
            from transformers.models.pixtral.configuration_pixtral import (
                PixtralVisionConfig,
            )
            from transformers.models.pixtral.modeling_pixtral import (
                PixtralRotaryEmbedding as TransformersPixtralEmb,
                apply_rotary_pos_emb,
                position_ids_in_meshgrid as hf_position_ids_in_meshgrid,
            )
        except ImportError:
            self.skipTest("Unable to import Transformer's Pixtral Model / Config")

        from fms.models.pixtral_vision import (
            get_positions_in_meshgrid as fms_get_positions_in_meshgrid,
        )

        # Configuration
        dim = 64
        ratio = 10_000.0
        image_size = 1024
        patch_size = 16
        batch_size = 1
        num_heads = 16
        patch_h, patch_w = 2, 3
        num_patches = patch_h * patch_w

        # Create sample inputs
        query_proj = torch.ones(
            batch_size, num_heads, num_patches, dim, dtype=torch.float32
        )
        key_proj = torch.ones(
            batch_size, num_heads, num_patches, dim, dtype=torch.float32
        )

        # HF position_ids (1D flattened)
        position_ids_hf = hf_position_ids_in_meshgrid(
            patch_embeds_list=[torch.rand((1024, patch_h, patch_w))],
            max_width=image_size // patch_size,
        )

        # FMS position_ids (2D with batch dimension)
        position_ids_fms = fms_get_positions_in_meshgrid(
            patch_embeds_list=[torch.rand((1024, patch_h, patch_w))],
        )

        ############ Get HF results
        hf_config = PixtralVisionConfig(
            **{
                "hidden_size": 1024,
                "head_dim": dim,
                "rope_theta": ratio,
                "image_size": image_size,
                "patch_size": patch_size,
            }
        )
        transformers_emb = TransformersPixtralEmb(hf_config)
        cos, sin = transformers_emb(query_proj, position_ids_hf)
        query_hf, key_hf = apply_rotary_pos_emb(
            query_proj, key_proj, cos, sin, unsqueeze_dim=0
        )

        ############ Get FMS results
        fms_emb = PixtralRotaryEmbedding(dim, ratio, image_size, patch_size)

        # In FMS, the query and key are viewed as [1, 1064, 16, 64], i.e,.
        # [bsz, num_patches, num_heads, head_dim] prior to invoking the rotational
        # embeddings, so we need to permute the inputs.
        query_fms_format = query_proj.transpose(1, 2)
        key_fms_format = key_proj.transpose(1, 2)

        query_fms, key_fms = fms_emb.adjusted_qk(
            query_fms_format, key_fms_format, position_ids_fms
        )

        # Convert FMS format back to HF format
        def permute_fms_to_hf(tensor):
            """
            Permute tensor from FMS RoPE format to HF RoPE format.
            FMS: [x0, y0, x1, y1, x2, y2, ..., x15, y15] (interleaved pairs)
            HF: [x0, x1, x2, ..., x15, y0, y1, y2, ..., y15] (split halves)
            """
            # [B, L, H, D/2]
            *batch_dims, head_dim = tensor.shape
            half_dim = head_dim // 2
            # Reshape to separate interleaved pairs
            paired = tensor.reshape(*batch_dims, half_dim, 2)
            # Split into first and second elements of each pair
            first_half = paired[..., 0]  # x0, x1, x2, ..., x15
            second_half = paired[..., 1]  # y0, y1, y2, ..., y15

            # Concatenate: first all x's, then all y's
            return torch.cat([first_half, second_half], dim=-1)

        adjusted_query_fms = permute_fms_to_hf(query_fms.transpose(1, 2))
        adjusted_key_fms = permute_fms_to_hf(key_fms.transpose(1, 2))

        # Compare results
        torch.testing.assert_close(adjusted_query_fms, query_hf, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(adjusted_key_fms, key_hf, rtol=1e-4, atol=1e-5)
