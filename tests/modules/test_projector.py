import math
import pytest
import torch
import torch.nn as nn

from fms.modules.projector import (
    SpeechProjector,
    SpeechProjectorConfig,
    QFormerSelfAttention,
    QFormerCrossAttention,
    QFormerAttentionOutput,
    QFormerLayer,
)


@pytest.fixture
def hf_aligned_config():
    """HF-aligned projector config matching granite-speech-3.3-8b."""
    return SpeechProjectorConfig(
        encoder_dim=1024,
        decoder_dim=4096,
        num_queries=3,
        window_size=15,
        num_hidden_layers=2,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
    )


@pytest.fixture
def small_config():
    """Smaller config for faster tests."""
    return SpeechProjectorConfig(
        encoder_dim=256,
        decoder_dim=512,
        num_queries=3,
        window_size=15,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_dropout_prob=0.0,
        attention_dropout_prob=0.0,
    )


class TestSpeechProjectorConfig:
    def test_default_config(self):
        config = SpeechProjectorConfig()
        assert config.encoder_dim == 1024
        assert config.decoder_dim == 2048
        assert config.num_queries == 3
        assert config.window_size == 15
        assert config.num_hidden_layers == 2

    def test_hf_aligned_config(self, hf_aligned_config):
        assert hf_aligned_config.num_queries == 3
        assert hf_aligned_config.window_size == 15
        downsample_rate = hf_aligned_config.window_size // hf_aligned_config.num_queries
        assert downsample_rate == 5


class TestWindowProcessing:
    def test_window_padding_exact(self, small_config):
        projector = SpeechProjector(small_config)
        projector.eval()

        seq_len = 15
        x = torch.randn(1, seq_len, small_config.encoder_dim)

        with torch.no_grad():
            output = projector(x)

        assert output.shape == (1, 3, small_config.decoder_dim)

    def test_window_padding_partial(self, small_config):
        projector = SpeechProjector(small_config)
        projector.eval()

        x = torch.randn(1, 14, small_config.encoder_dim)
        with torch.no_grad():
            output = projector(x)
        assert output.shape == (1, 3, small_config.decoder_dim)

        x = torch.randn(1, 16, small_config.encoder_dim)
        with torch.no_grad():
            output = projector(x)
        assert output.shape == (1, 6, small_config.decoder_dim)

    def test_window_output_shape_formula(self, small_config):
        projector = SpeechProjector(small_config)
        projector.eval()

        test_cases = [
            (14, 1, 3),
            (15, 1, 3),
            (16, 2, 6),
            (30, 2, 6),
            (45, 3, 9),
            (100, 7, 21),
        ]

        for seq_len, expected_nblocks, expected_queries in test_cases:
            x = torch.randn(1, seq_len, small_config.encoder_dim)
            with torch.no_grad():
                output = projector(x)

            actual_nblocks = math.ceil(seq_len / small_config.window_size)
            assert actual_nblocks == expected_nblocks, f"seq_len={seq_len}"
            assert output.shape[1] == expected_queries, f"seq_len={seq_len}"

    def test_batch_processing(self, small_config):
        projector = SpeechProjector(small_config)
        projector.eval()

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 30, small_config.encoder_dim)
            with torch.no_grad():
                output = projector(x)
            assert output.shape[0] == batch_size


class TestQueryInitialization:
    def test_query_shape(self, small_config):
        projector = SpeechProjector(small_config)
        assert projector.query_embeds.shape == (
            1, small_config.num_queries, small_config.encoder_dim
        )

    def test_query_normal_initialization(self, small_config):
        n_samples = 10
        all_queries = []

        for _ in range(n_samples):
            projector = SpeechProjector(small_config)
            all_queries.append(projector.query_embeds.data.flatten())

        all_queries = torch.cat(all_queries)

        mean = all_queries.mean().item()
        std = all_queries.std().item()

        assert abs(mean) < 0.1, f"Mean should be ~0, got {mean}"
        assert abs(std - 1.0) < 0.2, f"Std should be ~1, got {std}"

    def test_query_is_learnable(self, small_config):
        projector = SpeechProjector(small_config)
        x = torch.randn(1, 30, small_config.encoder_dim)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        assert projector.query_embeds.grad is not None
        assert not torch.isnan(projector.query_embeds.grad).any()


class TestQFormerComponents:
    def test_self_attention_output_shape(self, small_config):
        self_attn = QFormerSelfAttention(small_config)
        x = torch.randn(2, 3, small_config.encoder_dim)
        output = self_attn(x)
        assert output.shape == x.shape

    def test_cross_attention_output_shape(self, small_config):
        cross_attn = QFormerCrossAttention(small_config)
        queries = torch.randn(2, 3, small_config.encoder_dim)
        encoder_states = torch.randn(2, 15, small_config.encoder_dim)
        output = cross_attn(queries, encoder_states)
        assert output.shape == queries.shape

    def test_attention_output_residual(self, small_config):
        attn_out = QFormerAttentionOutput(small_config)
        hidden = torch.randn(2, 3, small_config.encoder_dim)
        residual = torch.randn(2, 3, small_config.encoder_dim)
        output = attn_out(hidden, residual)
        assert output.shape == hidden.shape
        assert not torch.allclose(output, hidden)
        assert not torch.allclose(output, residual)

    def test_qformer_layer_output_shape(self, small_config):
        layer = QFormerLayer(small_config)
        queries = torch.randn(2, 3, small_config.encoder_dim)
        encoder_states = torch.randn(2, 15, small_config.encoder_dim)
        output = layer(queries, encoder_states)
        assert output.shape == queries.shape


class TestGradientFlow:
    def test_gradient_to_input(self, small_config):
        projector = SpeechProjector(small_config)
        x = torch.randn(2, 30, small_config.encoder_dim, requires_grad=True)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_gradient_to_all_parameters(self, small_config):
        projector = SpeechProjector(small_config)
        x = torch.randn(2, 30, small_config.encoder_dim)
        output = projector(x)
        loss = output.sum()
        loss.backward()

        for name, param in projector.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestNumericalStability:
    def test_no_nan_output(self, small_config):
        projector = SpeechProjector(small_config)
        projector.eval()

        for seq_len in [14, 15, 16, 30, 100, 500]:
            x = torch.randn(2, seq_len, small_config.encoder_dim)
            with torch.no_grad():
                output = projector(x)
            assert not torch.isnan(output).any(), f"NaN at seq_len={seq_len}"

    def test_no_inf_output(self, small_config):
        projector = SpeechProjector(small_config)
        projector.eval()

        for seq_len in [14, 15, 16, 30, 100, 500]:
            x = torch.randn(2, seq_len, small_config.encoder_dim)
            with torch.no_grad():
                output = projector(x)
            assert not torch.isinf(output).any(), f"Inf at seq_len={seq_len}"

    def test_large_input_stability(self, small_config):
        projector = SpeechProjector(small_config)
        projector.eval()

        x = torch.randn(2, 30, small_config.encoder_dim) * 10
        with torch.no_grad():
            output = projector(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestCompressionRatio:
    def test_compression_ratio(self, hf_aligned_config):
        projector = SpeechProjector(hf_aligned_config)
        projector.eval()

        seq_len = 150
        x = torch.randn(1, seq_len, hf_aligned_config.encoder_dim)

        with torch.no_grad():
            output = projector(x)

        expected_output_len = (seq_len // hf_aligned_config.window_size) * hf_aligned_config.num_queries
        assert output.shape[1] == expected_output_len

        compression_ratio = seq_len / output.shape[1]
        assert compression_ratio == 5.0


@pytest.mark.slow
class TestHuggingFaceEquivalence:
    def test_output_shape_matches_hf(self, hf_aligned_config):
        projector = SpeechProjector(hf_aligned_config)
        projector.eval()

        for seq_len in [30, 45, 100, 500]:
            x = torch.randn(2, seq_len, hf_aligned_config.encoder_dim)
            with torch.no_grad():
                output = projector(x)

            nblocks = math.ceil(seq_len / hf_aligned_config.window_size)
            expected_queries = nblocks * hf_aligned_config.num_queries

            assert output.shape == (2, expected_queries, hf_aligned_config.decoder_dim), \
                f"seq_len={seq_len}, expected ({2}, {expected_queries}, {hf_aligned_config.decoder_dim}), got {output.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])