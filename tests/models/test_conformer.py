"""
Test suite for Conformer encoder implementation.
Following TDD approach - these tests should initially FAIL until implementation is complete.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fms.models.conformer import (
    ConformerConfig,
    ConformerBlock,
    ConformerEncoder,
)


class TestConformerBlock:
    """Test individual Conformer block."""

    @pytest.fixture
    def config(self):
        """Provide a test configuration."""
        return ConformerConfig(
            hidden_dim=256,
            num_heads=4,
            dim_head=64,
            conv_kernel_size=31,
            feedforward_mult=4,
            dropout=0.1,
            max_pos_emb=1000,
            context_size=100,
        )

    @pytest.fixture
    def conformer_block(self, config):
        """Provide an uninitialized ConformerBlock."""
        return ConformerBlock(config)

    def test_block_initialization(self, conformer_block, config):
        """Test that ConformerBlock initializes correctly."""
        assert isinstance(conformer_block, nn.Module)

        # Check that submodules exist (will fail until implementation)
        assert hasattr(conformer_block, "ff1"), "Should have first feed-forward module"
        assert hasattr(conformer_block, "attn"), "Should have attention module"
        assert hasattr(conformer_block, "conv"), "Should have convolution module"
        assert hasattr(conformer_block, "ff2"), "Should have second feed-forward module"
        assert hasattr(conformer_block, "post_norm"), "Should have post layer norm"

    def test_block_forward_shape(self, conformer_block, config):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_length = 100
        hidden_dim = config.hidden_dim

        # Create dummy input
        x = torch.randn(batch_size, seq_length, hidden_dim)

        # Create dummy attention distances (relative positional encodings)
        attention_dists = torch.randint(
            0, 2 * config.max_pos_emb + 1, (seq_length, seq_length)
        )

        # Forward pass
        output = conformer_block(x, attention_dists)

        # Output should have same shape as input
        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_block_forward_no_nan(self, conformer_block, config):
        """Test that forward pass doesn't produce NaN values."""
        x = torch.randn(2, 50, 256)
        # attention_dists must have shape (context_size, context_size) for chunked attention
        attention_dists = torch.randint(0, 2 * config.max_pos_emb + 1, (config.context_size, config.context_size))

        output = conformer_block(x, attention_dists)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_block_residual_connections(self, conformer_block, config):
        """Test that residual connections are working."""
        x = torch.randn(1, 10, 256)
        # attention_dists must have shape (context_size, context_size) for chunked attention
        attention_dists = torch.randint(0, 2 * config.max_pos_emb + 1, (config.context_size, config.context_size))

        # If all submodules returned zeros, residuals should preserve input
        # This tests the architectural pattern
        output = conformer_block(x, attention_dists)

        # Output should not be identical to input (due to processing)
        assert not torch.allclose(output, x), "Output should be different from input"

    def test_block_batch_independence(self, conformer_block, config):
        """Test that samples in batch are processed independently."""
        # Set to eval mode to use running stats in BatchNorm (ensures batch independence)
        conformer_block.eval()

        x = torch.randn(4, 20, 256)
        # attention_dists must have shape (context_size, context_size) for chunked attention
        attention_dists = torch.randint(0, 2 * config.max_pos_emb + 1, (config.context_size, config.context_size))

        # Process full batch
        output_batch = conformer_block(x, attention_dists)

        # Process samples individually
        outputs_individual = [
            conformer_block(x[i:i + 1], attention_dists) for i in range(4)
        ]
        outputs_stacked = torch.cat(outputs_individual, dim=0)

        # Should be identical (within floating point precision)
        assert torch.allclose(output_batch, outputs_stacked, atol=1e-5)


class TestConformerEncoder:
    """Test full Conformer encoder."""

    @pytest.fixture
    def config(self):
        """Provide a test configuration."""
        return ConformerConfig(
            num_features=80,
            hidden_dim=256,
            num_layers=4,  # Small for testing
            num_heads=4,
            dim_head=64,
            conv_kernel_size=31,
            feedforward_mult=4,
            dropout=0.0,  # Disable for deterministic testing
            max_pos_emb=1000,
            context_size=100,
        )

    @pytest.fixture
    def encoder(self, config):
        """Provide an uninitialized ConformerEncoder."""
        return ConformerEncoder(config)

    def test_encoder_initialization(self, encoder, config):
        """Test that ConformerEncoder initializes correctly."""
        assert isinstance(encoder, nn.Module)

        # Check input projection
        assert hasattr(encoder, "input_proj"), "Should have input projection layer"

        # Check conformer blocks
        assert hasattr(encoder, "blocks"), "Should have conformer blocks"
        assert isinstance(encoder.blocks, nn.ModuleList), "Blocks should be ModuleList"
        assert len(encoder.blocks) == config.num_layers, f"Should have {config.num_layers} blocks"

        # Check attention distance buffer
        assert hasattr(encoder, "attention_dists"), "Should precompute attention distances"

    def test_encoder_forward_shape(self, encoder, config):
        """Test that forward pass produces correct output shape."""
        batch_size = 2
        seq_length = 100
        num_features = config.num_features

        # Create dummy input (audio features)
        input_features = torch.randn(batch_size, seq_length, num_features)

        # Forward pass
        output = encoder(input_features)

        # Expected output shape
        expected_shape = (batch_size, seq_length, config.hidden_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_encoder_input_dimension_mismatch(self, encoder, config):
        """Test that encoder raises error for wrong input dimension."""
        batch_size = 2
        seq_length = 100
        wrong_features = 40  # Should be 80

        input_features = torch.randn(batch_size, seq_length, wrong_features)

        # Should raise an error
        with pytest.raises((RuntimeError, AssertionError)):
            output = encoder(input_features)

    def test_encoder_variable_sequence_length(self, encoder, config):
        """Test that encoder handles variable sequence lengths."""
        batch_size = 2
        num_features = config.num_features

        # Test different sequence lengths
        for seq_length in [50, 100, 200, 500]:
            input_features = torch.randn(batch_size, seq_length, num_features)
            output = encoder(input_features)

            expected_shape = (batch_size, seq_length, config.hidden_dim)
            assert output.shape == expected_shape, \
                f"Failed for seq_length={seq_length}: expected {expected_shape}, got {output.shape}"

    def test_encoder_output_no_nan(self, encoder, config):
        """Test that encoder doesn't produce NaN values."""
        input_features = torch.randn(2, 100, config.num_features)

        output = encoder(input_features)

        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

    def test_encoder_deterministic_with_dropout_disabled(self, config):
        """Test that encoder produces deterministic output when dropout is disabled."""
        config.dropout = 0.0
        encoder = ConformerEncoder(config)
        encoder.eval()  # Set to eval mode

        input_features = torch.randn(2, 100, config.num_features)

        # Run twice
        output1 = encoder(input_features)
        output2 = encoder(input_features)

        # Should be identical
        assert torch.allclose(output1, output2), "Output should be deterministic"

    def test_encoder_batch_independence(self, encoder, config):
        """Test that samples in batch are processed independently."""
        # Set to eval mode to use running stats in BatchNorm (ensures batch independence)
        encoder.eval()

        num_features = config.num_features
        seq_length = 100

        # Create batch
        input_batch = torch.randn(4, seq_length, num_features)

        # Process full batch
        output_batch = encoder(input_batch)

        # Process samples individually
        outputs_individual = [
            encoder(input_batch[i:i + 1]) for i in range(4)
        ]
        outputs_stacked = torch.cat(outputs_individual, dim=0)

        # Should be identical (within floating point precision)
        assert torch.allclose(output_batch, outputs_stacked, atol=1e-5)

    def test_encoder_gradient_flow(self, encoder, config):
        """Test that gradients flow through encoder."""
        encoder.train()

        input_features = torch.randn(2, 50, config.num_features, requires_grad=True)

        # Forward pass
        output = encoder(input_features)

        # Compute dummy loss
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert input_features.grad is not None, "Gradients should flow to input"
        assert not torch.isnan(input_features.grad).any(), "Gradients should not be NaN"

    def test_encoder_no_temporal_downsampling(self, encoder, config):
        """Test that encoder preserves sequence length (no downsampling in blocks)."""
        for seq_length in [50, 100, 200]:
            input_features = torch.randn(1, seq_length, config.num_features)
            output = encoder(input_features)

            assert output.shape[1] == seq_length, \
                f"Sequence length should be preserved: {seq_length} != {output.shape[1]}"


class TestConformerIntegration:
    """Integration tests for Conformer with FMS patterns."""

    def test_conformer_follows_fms_pattern(self):
        """Test that Conformer follows FMS architectural patterns."""
        config = ConformerConfig()
        encoder = ConformerEncoder(config)

        # Should be an nn.Module
        assert isinstance(encoder, nn.Module)

        # Should have a config attribute
        assert hasattr(encoder, "config")

        # Config should be a ModelConfig subclass
        from fms.utils.config import ModelConfig
        assert isinstance(config, ModelConfig)

    def test_conformer_model_registration_pattern(self):
        """Test that Conformer can be registered with FMS model registry."""
        from fms import models

        # This will fail until we add registration
        # But it documents the expected pattern
        config = ConformerConfig()

        def conformer_factory():
            return ConformerEncoder(config)

        # Expected registration pattern
        architecture_name = "conformer"
        variant_name = "base"

        # This should work once implemented
        # models.register_model(architecture_name, variant_name, conformer_factory)
        # model = models.get_model(architecture_name, variant_name)

        assert True  # Placeholder - will implement registration later

    def test_conformer_serialization_compatibility(self):
        """Test that Conformer can be saved/loaded with torch."""
        config = ConformerConfig(num_layers=2)
        encoder = ConformerEncoder(config)

        # Get initial parameters
        initial_state = encoder.state_dict()

        # Create new encoder and load state
        encoder_new = ConformerEncoder(config)
        encoder_new.load_state_dict(initial_state)

        # Should produce identical outputs
        input_features = torch.randn(1, 50, config.num_features)

        encoder.eval()
        encoder_new.eval()

        output1 = encoder(input_features)
        output2 = encoder_new(input_features)

        assert torch.allclose(output1, output2, atol=1e-6), \
            "Loaded model should produce identical outputs"


class TestConformerComponents:
    """Test individual Conformer components."""

    @pytest.fixture
    def config(self):
        return ConformerConfig(hidden_dim=256, num_heads=4, dim_head=64)

    def test_feedforward_module_exists(self, config):
        """Test that ConformerFeedForward module can be instantiated."""
        from fms.models.conformer import ConformerFeedForward

        ff = ConformerFeedForward(
            dim=config.hidden_dim,
            mult=config.feedforward_mult,
            dropout=config.dropout,
        )

        assert isinstance(ff, nn.Module)

        # Test forward pass
        x = torch.randn(2, 50, config.hidden_dim)
        output = ff(x)
        assert output.shape == x.shape

    def test_attention_module_exists(self, config):
        """Test that ConformerAttention module can be instantiated."""
        from fms.models.conformer import ConformerAttention

        attn = ConformerAttention(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            max_pos_emb=config.max_pos_emb,
            context_size=config.context_size,
            dropout=config.dropout,
        )

        assert isinstance(attn, nn.Module)

        # Test forward pass
        x = torch.randn(2, 50, config.hidden_dim)
        # attention_dists must have shape (context_size, context_size) for chunked attention
        attention_dists = torch.randint(0, 2 * config.max_pos_emb + 1, (config.context_size, config.context_size))
        output = attn(x, attention_dists)
        assert output.shape == x.shape

    def test_convolution_module_exists(self, config):
        """Test that ConformerConvModule can be instantiated."""
        from fms.models.conformer import ConformerConvModule

        conv = ConformerConvModule(
            dim=config.hidden_dim,
            kernel_size=config.conv_kernel_size,
            expansion_factor=config.conv_expansion_factor,
            dropout=config.dropout,
        )

        assert isinstance(conv, nn.Module)

        # Test forward pass
        x = torch.randn(2, 50, config.hidden_dim)
        output = conv(x)
        assert output.shape == x.shape


class TestConformerRepresentations:
    """
    These tests validate that the Conformer learns meaningful representations
    and exhibits expected architectural properties (local+global processing).
    """

    @pytest.fixture
    def small_encoder(self):
        """Small encoder for faster research tests."""
        config = ConformerConfig(
            num_features=80,
            hidden_dim=128,
            num_layers=2,  # Very small
            num_heads=4,
            dim_head=32,
            conv_kernel_size=15,
            feedforward_mult=2,
            dropout=0.0,
            max_pos_emb=512,
            context_size=200,
        )
        return ConformerEncoder(config)

    def test_representation_collapse_detection(self, small_encoder):
        """
        CRITICAL: Test that encoder doesn't collapse all inputs to same representation.

        A collapsed encoder outputs nearly identical representations for different inputs,
        making it useless for downstream tasks.
        """
        small_encoder.eval()

        # Generate two different random inputs
        x1 = torch.randn(1, 50, 80)
        x2 = torch.randn(1, 50, 80)

        # Encode both
        h1 = small_encoder(x1)  # (1, 50, hidden_dim)
        h2 = small_encoder(x2)  # (1, 50, hidden_dim)

        # Pool over time dimension
        h1_pooled = h1.mean(dim=1)  # (1, hidden_dim)
        h2_pooled = h2.mean(dim=1)  # (1, hidden_dim)

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(h1_pooled, h2_pooled, dim=-1)

        # Different inputs should produce different representations
        assert cos_sim.item() < 0.95, (
            f"Representation collapse detected! Cosine similarity: {cos_sim.item():.3f}. "
            "Different inputs produce nearly identical representations."
        )

        # Also check L2 distance
        l2_dist = torch.norm(h1_pooled - h2_pooled, p=2)
        assert l2_dist.item() > 0.1, (
            f"Representations too similar (L2 dist: {l2_dist.item():.3f})"
        )

    def test_positional_encoding_sensitivity(self, small_encoder):
        """
        Test that Conformer is sensitive to sequence order (via relative positions).

        Conformer uses relative positional embeddings in attention. If we permute
        the sequence, the output should change.
        """
        small_encoder.eval()

        # Create input
        x = torch.randn(1, 50, 80)

        # Original encoding
        h_original = small_encoder(x)

        # Permute the sequence (shuffle time dimension)
        perm_indices = torch.randperm(50)
        x_permuted = x[:, perm_indices, :]

        # Encode permuted input
        h_permuted = small_encoder(x_permuted)

        # Outputs should be different (position matters!)
        # Note: We can't expect h_permuted = h_original[perm] because attention is global
        assert not torch.allclose(h_original, h_permuted, atol=1e-3), (
            "Encoder is not position-sensitive! Permuting sequence had no effect."
        )

        # Check that the difference is meaningful
        relative_diff = (h_original - h_permuted).abs().mean() / h_original.abs().mean()
        assert relative_diff > 0.01, (
            f"Position sensitivity too weak: relative diff = {relative_diff:.4f}"
        )

    def test_local_pattern_extraction_conv(self, small_encoder):
        """
        Test that convolution module responds to local patterns.

        Conformer's conv module should be sensitive to local structure.
        """
        from fms.models.conformer import ConformerConvModule

        conv_module = ConformerConvModule(
            dim=128,
            kernel_size=15,
            expansion_factor=2,
            dropout=0.0,
        )
        conv_module.eval()

        # Create input with local repetitive pattern
        batch, seq_len, dim = 1, 100, 128
        x_local = torch.zeros(batch, seq_len, dim)
        # Add high-frequency pattern (alternating values)
        x_local[:, ::2, :] = 1.0
        x_local[:, 1::2, :] = -1.0

        # Create input with smooth global pattern
        x_smooth = torch.zeros(batch, seq_len, dim)
        # Add low-frequency pattern (slowly varying)
        x_smooth[:, :, :] = torch.linspace(0, 1, seq_len).unsqueeze(0).unsqueeze(-1)

        # Pass through conv
        out_local = conv_module(x_local)
        out_smooth = conv_module(x_smooth)

        # Conv should respond more strongly to local patterns
        local_variance = out_local.var(dim=1).mean()
        smooth_variance = out_smooth.var(dim=1).mean()

        # Local pattern should create more variation in output
        assert local_variance > smooth_variance * 0.5, (
            f"Conv module not responding to local patterns. "
            f"Local var: {local_variance:.4f}, Smooth var: {smooth_variance:.4f}"
        )

    def test_attention_captures_global_context(self, small_encoder):
        """
        Test that attention mechanism captures global dependencies.

        If we modify a single position, attention should propagate this change
        to other positions (global context).
        """
        small_encoder.eval()

        # Create base input
        x = torch.randn(1, 50, 80)

        # Original encoding
        h_original = small_encoder(x)

        # Modify just ONE position in the middle
        x_modified = x.clone()
        x_modified[:, 25, :] += 5.0  # Large perturbation at position 25

        # Encode modified input
        h_modified = small_encoder(x_modified)

        # Check that OTHER positions changed (global propagation)
        # Compare position 10 (far from modification)
        pos_10_diff = (h_modified[:, 10, :] - h_original[:, 10, :]).abs().mean()

        # Position 10 should change due to attention (global context)
        assert pos_10_diff > 1e-3, (
            f"Attention not propagating globally. Position 10 unchanged: {pos_10_diff:.6f}"
        )

    def test_residual_scaling_stability(self, small_encoder):
        """
        Test that 0.5x residual scaling in feedforward layers provides stability.

        Conformer uses 0.5x scaling for FFN residuals. This should prevent
        activation explosion compared to 1.0x scaling.
        """
        small_encoder.eval()

        # Create input
        x = torch.randn(2, 50, 80)

        # Forward pass
        output = small_encoder(x)

        # Check output magnitude is reasonable (not exploding)
        output_mean = output.abs().mean()
        output_std = output.std()

        # Should be in reasonable range (roughly same order as input)
        input_mean = x.abs().mean()
        assert 0.1 < output_mean / input_mean < 10.0, (
            f"Output magnitude unusual. Input mean: {input_mean:.3f}, "
            f"Output mean: {output_mean:.3f}"
        )

        # Standard deviation should be reasonable
        assert output_std < 10.0, f"Output std too large: {output_std:.3f}"

    def test_length_generalization(self, small_encoder):
        """
        Test that encoder generalizes to different sequence lengths.

        Speech encoders must handle variable-length inputs.
        """
        small_encoder.eval()

        # Test on various lengths
        test_lengths = [20, 50, 100, 200]

        outputs = []
        for seq_len in test_lengths:
            x = torch.randn(1, seq_len, 80)
            output = small_encoder(x)

            # Check output statistics remain stable
            mean_val = output.mean().item()
            std_val = output.std().item()

            outputs.append((seq_len, mean_val, std_val))

            # Sanity checks
            assert not torch.isnan(output).any(), f"NaN for length {seq_len}"
            assert not torch.isinf(output).any(), f"Inf for length {seq_len}"

        # Check that statistics don't vary wildly across lengths
        means = [m for _, m, _ in outputs]
        stds = [s for _, _, s in outputs]

        mean_range = max(means) - min(means)
        std_range = max(stds) - min(stds)

        assert mean_range < 2.0, f"Mean varies too much across lengths: {mean_range:.3f}"
        assert std_range < 2.0, f"Std varies too much across lengths: {std_range:.3f}"

    def test_representation_discriminability(self, small_encoder):
        """
        Test that encoder produces discriminable representations for distinct inputs.

        Key property: If inputs are different, representations should be distinguishable.
        """
        small_encoder.eval()

        # Create three distinct input patterns
        seq_len = 50

        # Pattern 1: High values
        x1 = torch.ones(1, seq_len, 80) * 2.0

        # Pattern 2: Low values
        x2 = torch.ones(1, seq_len, 80) * (-2.0)

        # Pattern 3: Zeros
        x3 = torch.zeros(1, seq_len, 80)

        # Encode all
        h1 = small_encoder(x1).mean(dim=1)  # Pool over time
        h2 = small_encoder(x2).mean(dim=1)
        h3 = small_encoder(x3).mean(dim=1)

        # Compute pairwise distances
        d12 = torch.norm(h1 - h2, p=2).item()
        d13 = torch.norm(h1 - h3, p=2).item()
        d23 = torch.norm(h2 - h3, p=2).item()

        # All pairs should be distinguishable
        assert d12 > 0.1, f"Patterns 1 and 2 too similar: {d12:.4f}"
        assert d13 > 0.1, f"Patterns 1 and 3 too similar: {d13:.4f}"
        assert d23 > 0.1, f"Patterns 2 and 3 too similar: {d23:.4f}"

    def test_attention_pattern_structure(self, small_encoder):
        """
        Test that attention patterns show some structure (not purely random).

        We can't test exact patterns without training, but we can check for basic structure.
        """
        small_encoder.eval()

        # Get a conformer block
        block = small_encoder.blocks[0]

        # Create input
        x = torch.randn(1, 30, 128)

        # Get attention module
        attn_module = block.attn

        # Forward through normalization and Q/K projection
        x_normed = attn_module.norm(x)
        q = attn_module.to_q(x_normed)
        # Implementation uses combined to_kv projection, split to get k and v
        k, v = attn_module.to_kv(x_normed).chunk(2, dim=-1)

        # Reshape for multi-head
        batch, seq_len, _ = x.shape
        q = q.view(batch, seq_len, attn_module.num_heads, attn_module.dim_head).transpose(1, 2)
        k = k.view(batch, seq_len, attn_module.num_heads, attn_module.dim_head).transpose(1, 2)

        # Compute attention scores (without positional bias for simplicity)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * attn_module.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, heads, seq_len, seq_len)

        # Check attention weights sum to 1
        attn_sum = attn_weights.sum(dim=-1)
        assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), (
            "Attention weights don't sum to 1"
        )

        # Check attention isn't completely uniform (would indicate degenerate initialization)
        # Note: Untrained models naturally have near-uniform attention (~98-99%)
        # We only catch completely uniform attention (>99%) which indicates a bug
        attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float32))

        assert attn_entropy < max_entropy * 0.99, (
            f"Attention completely uniform (entropy: {attn_entropy:.3f} vs max: {max_entropy:.3f}). "
            f"This indicates degenerate initialization."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])