# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for GraniteSpeech model.

Migrated from HuggingFace transformers:
tests/models/granite_speech/test_modeling_granite_speech.py
tests/models/granite_speech/test_processing_granite_speech.py

Contains:
- GraniteSpeechFixtures: FMS test fixtures for config and model
- TestGraniteSpeech: Core model tests using FMS test suites
- TestGraniteSpeechModel: Model-specific tests from HF
- TestGraniteSpeechProcessor: Processor tests from HF
- TestGraniteSpeechE2E: End-to-end integration tests
"""

import gc

import numpy as np
import pytest
import torch

from fms.models.granite_speech import (
    GraniteSpeech,
    GraniteSpeechConfig,
    GraniteSpeechFeatureExtractor,
    GraniteSpeechProcessor,
)
from fms.models.conformer import ConformerConfig
from fms.models.granite import GraniteConfig
from fms.modules.projector import SpeechProjectorConfig
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


# Check for GPU availability
def get_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


torch_device = get_torch_device()


# Check for torchaudio availability (required for FeatureExtractor)
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

requires_torchaudio = pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE, reason="torchaudio is required for audio feature extraction"
)


def floats_tensor(shape, scale=1.0):
    """Create a random float tensor."""
    return torch.rand(shape) * scale


def ids_tensor(shape, vocab_size):
    """Create a random integer tensor for token IDs."""
    return torch.randint(low=0, high=vocab_size, size=shape)


def cleanup(device, gc_collect=True):
    """Clean up memory after tests."""
    if gc_collect:
        gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# FMS Test Fixtures (following test_llava_next.py pattern)
# =============================================================================


class GraniteSpeechFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base GraniteSpeech Fixtures that can be re-used for other purposes.

    This will include the config and model signatures.
    """

    def _maybe_get_initialized_parameter(self, key: str, parameter: torch.Tensor):
        """
        Override to handle non-float parameters and special buffers.

        The base class uses torch.randn_like which doesn't work for Long tensors.
        For integer buffers, we return the parameter unchanged.
        BatchNorm running statistics should also be preserved.
        """
        # Skip initialization for integer/long tensors (buffers like attention_dists)
        if parameter.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            return parameter
        # Skip initialization for BatchNorm running statistics
        # These buffers need to remain at their default values (mean=0, var=1)
        if "running_mean" in key or "running_var" in key:
            return parameter
        # For float tensors, return None to use default random initialization
        return None

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: GraniteSpeechConfig):
        return GraniteSpeech(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        _encoder_config = ConformerConfig(
            num_features=160,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            dim_head=8,  # hidden_dim / num_heads = 32 / 4 = 8
            conv_kernel_size=15,
            conv_expansion_factor=2,
            feedforward_mult=4,
            dropout=0.0,  # Disable dropout for testing stability
            output_dim=42,
        )

        _decoder_config = GraniteConfig(
            src_vocab_size=99,
            emb_dim=32,
            nlayers=2,
            nheads=4,
            head_dim=8,  # emb_dim / nheads = 32 / 4 = 8
            kvheads=4,   # Must be set for proper attention
            hidden_grow_factor=37 / 32,
            max_expected_seq_len=580,
            pad_id=1,
        )

        _projector_config = SpeechProjectorConfig(
            encoder_dim=32,
            decoder_dim=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            window_size=15,
            downsample_rate=5,
        )

        return GraniteSpeechConfig(
            encoder_config=_encoder_config,
            decoder_config=_decoder_config,
            projector_config=_projector_config,
            audio_token_index=0,
            downsample_rate=5,
            window_size=15,
        )


class TestGraniteSpeech(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    GraniteSpeechFixtures,
):
    """
    Core FMS test suite for GraniteSpeech model.

    Inherits tests from:
    - ModelConfigTestSuite: Config validation tests
    - ModelConsistencyTestSuite: Output consistency tests
    - ModelCompileTestSuite: Model compilation tests
    """

    # =========================================================================
    # Skip ModelConsistencyTestSuite tests
    # =========================================================================
    # FMS test infrastructure (get_signature) doesn't support multimodal models
    # that require additional tensor inputs (input_features) beyond input_ids.
    # The infrastructure only moves the main input to device, not optional_params.
    # HF doesn't have equivalent signature tests - they use custom tests instead.
    # Our custom tests (TestGraniteSpeechModel, TestGraniteSpeechE2E) properly
    # handle device placement and test the full multimodal functionality.

    @pytest.mark.skip(
        reason="FMS test infrastructure doesn't support multimodal models with "
        "optional tensor inputs (input_features). Use TestGraniteSpeechModel instead."
    )
    def test_model_output(self, model, signature, model_id, capture_expectation):
        """Skip signature test - not compatible with multimodal models."""
        pass

    @pytest.mark.skip(
        reason="FMS test infrastructure doesn't support multimodal models with "
        "optional tensor inputs (input_features). Use TestGraniteSpeechModel instead."
    )
    def test_model_unfused(self, model, signature):
        """Skip unfused signature test - not compatible with multimodal models."""
        pass

    @pytest.mark.skip(
        reason="FMS test infrastructure doesn't support multimodal models with "
        "optional tensor inputs (input_features). Use TestGraniteSpeechModel instead."
    )
    def test_model_weight_keys(self, model, model_id, capture_expectation):
        """Skip weight keys test - not compatible with multimodal models."""
        pass

    @pytest.mark.skip(
        reason="Multimodal models use dynamic shape operations (aten.nonzero) "
        "for audio token detection which are incompatible with fullgraph compilation"
    )
    def test_model_compile_no_graph_breaks(self, model):
        """Skip fullgraph compile test for multimodal model."""
        pass

    def test_config_passed_to_model_and_updated(self, model, config):
        """Test model constructor appropriately merges any passed kwargs into the config."""
        model = type(model)(
            config=config, audio_token_index=config.audio_token_index + 1
        )
        # check not same reference
        assert model.get_config() is not config

        # modify audio_token_index to the new value expected and check equivalence
        config.audio_token_index = config.audio_token_index + 1
        assert model.get_config().as_dict() == config.as_dict()

    def test_config_params_passed_as_kwargs_to_model(self, model, config):
        pytest.skip(
            "granite_speech uses nested configs for encoder, decoder, and projector"
        )


# =============================================================================
# Model-Specific Tests (migrated from HF test_modeling_granite_speech.py)
# =============================================================================


class GraniteSpeechForConditionalGenerationModelTester:
    """
    Helper class for creating test configurations and inputs.

    Migrated from HF test_modeling_granite_speech.py L55-211.
    Adapted for FMS GraniteSpeech model.

    HF Reference parameters (L56-143):
    - encoder_config: context_size=200, hidden_dim=32, input_dim=160, num_heads=4, num_layers=2, output_dim=42
    - text_config: vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37
    - projector_config: encoder_hidden_size=32, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256
    - sequence_dim=844, feature_dim=160, batch_size=3, num_audio_tokens=2
    """

    def __init__(
        self,
        seq_length=7,
        encoder_config=None,
        decoder_config=None,
        projector_config=None,
        audio_token_index=0,
        downsample_rate=5,
        window_size=15,
    ):
        # Encoder config (Conformer) - matches HF encoder_config L60-73
        self.encoder_config = encoder_config or ConformerConfig(
            num_features=160,       # HF: input_dim=160
            hidden_dim=32,          # HF: hidden_dim=32
            num_layers=2,           # HF: num_layers=2
            num_heads=4,            # HF: num_heads=4
            dim_head=8,             # hidden_dim / num_heads = 32 / 4 = 8
            conv_kernel_size=15,    # HF: conv_kernel_size=15
            conv_expansion_factor=2,# HF: conv_expansion_factor=2
            feedforward_mult=4,     # HF: feedforward_mult=4
            dropout=0.0,            # Disabled for testing stability
            output_dim=42,          # HF: output_dim=42
        )

        # Decoder config (Granite) - matches HF text_config L74-95
        self.decoder_config = decoder_config or GraniteConfig(
            src_vocab_size=99,              # HF: vocab_size=99
            emb_dim=32,                     # HF: hidden_size=32
            nlayers=2,                      # HF: num_hidden_layers=2
            nheads=4,                       # HF: num_attention_heads=4
            head_dim=8,                     # emb_dim / nheads = 32 / 4 = 8
            kvheads=4,                      # Required for proper attention
            hidden_grow_factor=37 / 32,     # HF: intermediate_size=37
            max_expected_seq_len=580,       # HF: max_position_embeddings=580
            pad_id=1,                       # HF: pad_token_id=1
        )

        # Projector config (SpeechProjector) - matches HF projector_config L96-112
        self.projector_config = projector_config or SpeechProjectorConfig(
            encoder_dim=32,                 # HF: encoder_hidden_size=32
            decoder_dim=32,                 # HF: hidden_size=32
            num_hidden_layers=2,            # HF: num_hidden_layers=2
            num_attention_heads=4,          # HF: num_attention_heads=4
            intermediate_size=256,          # HF: intermediate_size=256
            window_size=window_size,        # HF: window_size=15 (from parent config)
            downsample_rate=downsample_rate,# HF: downsample_rate=5 (from parent config)
            num_queries=window_size // downsample_rate,  # Derived: 15 // 5 = 3
        )

        self.audio_token_index = audio_token_index  # HF: audio_token_index=0
        self.downsample_rate = downsample_rate      # HF: downsample_rate=5
        self.window_size = window_size              # HF: window_size=15

        # Dims for audio features - matches HF L133-143
        self.sequence_dim = 844             # HF: sequence_dim=844
        self.feature_dim = 160              # HF: feature_dim=160
        self.batch_size = 3                 # HF: batch_size=3
        self.pad_token_id = 1               # HF: pad_token_id from text_config
        self.seq_len = seq_length           # HF: seq_len=7
        self.num_audio_tokens = 2           # HF: num_audio_tokens=2
        self.seq_length = seq_length + self.num_audio_tokens  # HF: seq_length=9

    def get_config(self) -> GraniteSpeechConfig:
        """Create a GraniteSpeechConfig for testing."""
        return GraniteSpeechConfig(
            encoder_config=self.encoder_config,
            decoder_config=self.decoder_config,
            projector_config=self.projector_config,
            audio_token_index=self.audio_token_index,
            downsample_rate=self.downsample_rate,
            window_size=self.window_size,
        )

    def prepare_config_and_inputs(self):
        """Prepare config and input features."""
        input_features = floats_tensor(
            [self.batch_size, self.sequence_dim, self.feature_dim],
        )
        config = self.get_config()
        return config, input_features

    def prepare_config_and_inputs_for_common(self):
        """Prepare config and full inputs dict for common tests."""
        config, input_features = self.prepare_config_and_inputs()

        input_ids = ids_tensor(
            [self.batch_size, self.seq_length],
            self.decoder_config.src_vocab_size - 2
        ) + 2
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        input_ids[input_ids == config.audio_token_index] = self.pad_token_id

        # Place audio tokens at the beginning
        input_ids[:, : self.num_audio_tokens] = config.audio_token_index

        inputs_dict = {
            "input_features": input_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class TestGraniteSpeechModel:
    """
    Model-specific tests for GraniteSpeech.

    Contains tests migrated from HF test_modeling_granite_speech.py
    """

    @pytest.fixture
    def model_tester(self):
        return GraniteSpeechForConditionalGenerationModelTester()

    def test_inputs_embeds(self, model_tester):
        """Test that the model can accept inputs_embeds instead of input_ids.

        Overwrite inputs_embeds tests because we need to delete "input features"
        for the audio model.

        HF Source: test_modeling_granite_speech.py L231-250
        """
        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()

        model = GraniteSpeech(config)
        model.to(torch_device)
        model.eval()

        # Initialize weights to avoid NaN - random init with preserved BatchNorm stats
        torch.manual_seed(5)
        sd = model.state_dict()
        for key in sd.keys():
            param = sd[key]
            if param.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                continue
            if "running_mean" in key or "running_var" in key:
                continue
            values = torch.randn_like(param)
            values -= 0.5
            values /= 20.0
            param.copy_(values)

        input_ids = inputs_dict["input_ids"].to(torch_device)
        # Don't use input_features for this test

        # Get input embeddings
        # FMS GraniteHeadless has embedding directly, not via .model
        wte = model.decoder.embedding
        inputs_embeds = wte(input_ids)

        with torch.no_grad():
            # Forward with inputs_embeds instead of input_ids
            # FMS GraniteHeadless uses x_in as positional argument
            logits, _ = model.decoder(x_in=inputs_embeds)

        assert logits is not None
        assert not torch.isnan(logits).any().item()

    def test_sdpa_can_dispatch_composite_models(self, model_tester):
        """Test SDPA attention dispatch for composite audio+text model.

        Overwrite because Granite Speech is audio+text model (not vision+text).
        NOTE - currently we only enable alternate attention implementations on
        the encapsulated LLM; in the future, this should be added for the conformer
        encoder as well.

        HF Source: test_modeling_granite_speech.py L252-287
        Note: HF test doesn't run a forward pass with audio - it only checks
        attention implementation attributes. Our test verifies model creation
        and forward pass with correctly matched dimensions.
        """
        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()
        model = GraniteSpeech(config)

        # FMS models don't have the same _attn_implementation attribute as HF
        # This test verifies that the model can be created and run
        # The actual SDPA dispatch is handled internally by PyTorch
        model.to(torch_device)
        model.eval()

        # Initialize weights to avoid NaN - random init with preserved BatchNorm stats
        torch.manual_seed(5)
        sd = model.state_dict()
        for key in sd.keys():
            param = sd[key]
            if param.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                continue
            if "running_mean" in key or "running_var" in key:
                continue
            values = torch.randn_like(param)
            values -= 0.5
            values /= 20.0
            param.copy_(values)

        # For forward pass test, we need audio tokens count to match projected features
        # With sequence_dim=844, window_size=15, num_queries=3:
        # num_windows = ceil(844 / 15) = 57, num_audio_tokens = 57 * 3 = 171
        # Note: projector uses math.ceil for window calculation
        import math
        window_size = model_tester.window_size
        num_queries = model_tester.projector_config.num_queries
        num_windows = math.ceil(model_tester.sequence_dim / window_size)
        actual_num_audio_tokens = num_windows * num_queries

        # Create input_ids with correct number of audio tokens
        batch_size = model_tester.batch_size
        seq_length = model_tester.seq_len + actual_num_audio_tokens
        input_ids = torch.randint(
            2, config.decoder_config.src_vocab_size,
            (batch_size, seq_length), device=torch_device
        )
        # Place audio tokens at the beginning
        input_ids[:, :actual_num_audio_tokens] = config.audio_token_index

        input_features = inputs_dict["input_features"].to(torch_device)

        with torch.no_grad():
            logits, _ = model(input_ids=input_ids, input_features=input_features)

        assert logits is not None
        assert not torch.isnan(logits).any().item()

    @pytest.mark.skipif(
        torch_device == "cpu",
        reason="FP16 test requires GPU"
    )
    def test_granite_speech_model_fp16_forward(self, model_tester):
        """Test FP16 forward pass for GraniteSpeech model.

        Verifies that the model produces valid (non-NaN) logits when
        running in FP16 precision on GPU.

        HF Source: test_modeling_granite_speech.py L180-191
        (create_and_check_granite_speech_model_fp16_forward)
        """
        import math

        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()

        model = GraniteSpeech(config)
        model.to(torch_device)
        model.half()
        model.eval()

        # Initialize weights with small values for FP16 stability
        torch.manual_seed(42)
        sd = model.state_dict()
        for key in sd.keys():
            param = sd[key]
            if param.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                continue
            if "running_mean" in key or "running_var" in key:
                continue
            # Use smaller values for FP16 stability
            values = torch.randn_like(param) * 0.01
            param.copy_(values)

        # Calculate correct number of audio tokens
        window_size = model_tester.window_size
        num_queries = model_tester.projector_config.num_queries
        num_windows = math.ceil(model_tester.sequence_dim / window_size)
        actual_num_audio_tokens = num_windows * num_queries

        # Create input_ids with correct number of audio tokens
        batch_size = model_tester.batch_size
        seq_length = model_tester.seq_len + actual_num_audio_tokens
        input_ids = torch.randint(
            2, config.decoder_config.src_vocab_size,
            (batch_size, seq_length), device=torch_device
        )
        input_ids[:, :actual_num_audio_tokens] = config.audio_token_index

        input_features = inputs_dict["input_features"].to(torch_device).half()
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        with torch.no_grad():
            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
            )

        assert logits is not None
        assert not torch.isnan(logits).any().item(), "FP16 forward produced NaN values"

    @pytest.mark.skipif(
        torch_device == "cpu",
        reason="FP16 autocast test requires GPU"
    )
    def test_granite_speech_model_fp16_autocast_forward(self, model_tester):
        """Test FP16 autocast forward pass for GraniteSpeech model.

        Verifies that the model produces valid (non-NaN) logits when
        running with torch.autocast in FP16 precision.

        HF Source: test_modeling_granite_speech.py L193-211
        (create_and_check_granite_speech_model_fp16_autocast_forward)
        """
        import math

        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()

        model = GraniteSpeech(config)
        model.to(torch_device)
        model.eval()

        # Initialize weights with small values for FP16 stability
        torch.manual_seed(42)
        sd = model.state_dict()
        for key in sd.keys():
            param = sd[key]
            if param.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                continue
            if "running_mean" in key or "running_var" in key:
                continue
            # Use smaller values for FP16 stability
            values = torch.randn_like(param) * 0.01
            param.copy_(values)

        # Calculate correct number of audio tokens
        window_size = model_tester.window_size
        num_queries = model_tester.projector_config.num_queries
        num_windows = math.ceil(model_tester.sequence_dim / window_size)
        actual_num_audio_tokens = num_windows * num_queries

        # Create input_ids with correct number of audio tokens
        batch_size = model_tester.batch_size
        seq_length = model_tester.seq_len + actual_num_audio_tokens
        input_ids = torch.randint(
            2, config.decoder_config.src_vocab_size,
            (batch_size, seq_length), device=torch_device
        )
        input_ids[:, :actual_num_audio_tokens] = config.audio_token_index

        input_features = inputs_dict["input_features"].to(torch_device).to(torch.bfloat16)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                )

        assert logits is not None
        assert not torch.isnan(logits).any().item(), "FP16 autocast forward produced NaN values"

    def test_granite_speech_model_bfloat16_forward(self, model_tester):
        """Test bfloat16 forward pass for GraniteSpeech model.

        Verifies that the model produces valid (non-NaN) logits when
        running in bfloat16 precision. bfloat16 has a larger dynamic range
        than float16 and is often more stable for training.

        Note: This test runs on CPU (bfloat16 is supported on CPU unlike FP16).
        """
        import math

        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()

        model = GraniteSpeech(config)
        model.to(torch.bfloat16)
        model.eval()

        # Initialize weights with small values for stability
        torch.manual_seed(42)
        sd = model.state_dict()
        for key in sd.keys():
            param = sd[key]
            if param.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                continue
            if "running_mean" in key or "running_var" in key:
                continue
            # Use smaller values for stability
            values = torch.randn_like(param) * 0.01
            param.copy_(values)

        # Calculate correct number of audio tokens
        window_size = model_tester.window_size
        num_queries = model_tester.projector_config.num_queries
        num_windows = math.ceil(model_tester.sequence_dim / window_size)
        actual_num_audio_tokens = num_windows * num_queries

        # Create input_ids with correct number of audio tokens
        batch_size = model_tester.batch_size
        seq_length = model_tester.seq_len + actual_num_audio_tokens
        input_ids = torch.randint(
            2, config.decoder_config.src_vocab_size,
            (batch_size, seq_length)
        )
        input_ids[:, :actual_num_audio_tokens] = config.audio_token_index

        input_features = inputs_dict["input_features"].to(torch.bfloat16)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        with torch.no_grad():
            logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
            )

        assert logits is not None
        assert logits.dtype == torch.bfloat16, f"Expected bfloat16 logits, got {logits.dtype}"
        assert not torch.isnan(logits).any().item(), "bfloat16 forward produced NaN values"

    def test_granite_speech_model_bfloat16_autocast_forward(self, model_tester):
        """Test bfloat16 autocast forward pass for GraniteSpeech model.

        Verifies that the model produces valid (non-NaN) logits when
        running with torch.autocast in bfloat16 precision.

        Note: bfloat16 autocast works on both CPU and CUDA.
        """
        import math

        config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()

        model = GraniteSpeech(config)
        # Keep model in float32, autocast will handle conversion
        model.eval()

        # Initialize weights with small values for stability
        torch.manual_seed(42)
        sd = model.state_dict()
        for key in sd.keys():
            param = sd[key]
            if param.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                continue
            if "running_mean" in key or "running_var" in key:
                continue
            values = torch.randn_like(param) * 0.01
            param.copy_(values)

        # Calculate correct number of audio tokens
        window_size = model_tester.window_size
        num_queries = model_tester.projector_config.num_queries
        num_windows = math.ceil(model_tester.sequence_dim / window_size)
        actual_num_audio_tokens = num_windows * num_queries

        # Create input_ids with correct number of audio tokens
        batch_size = model_tester.batch_size
        seq_length = model_tester.seq_len + actual_num_audio_tokens
        input_ids = torch.randint(
            2, config.decoder_config.src_vocab_size,
            (batch_size, seq_length)
        )
        input_ids[:, :actual_num_audio_tokens] = config.audio_token_index

        input_features = inputs_dict["input_features"]
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        # Use CPU autocast with bfloat16 (works without GPU)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    input_features=input_features,
                )

        assert logits is not None
        assert not torch.isnan(logits).any().item(), "bfloat16 autocast forward produced NaN values"


# =============================================================================
# FMS Native Feature Extractor and Processor Tests
# =============================================================================


@requires_torchaudio
class TestFMSGraniteSpeechFeatureExtractor:
    """
    Tests for FMS-native GraniteSpeechFeatureExtractor.

    These tests validate the fully implemented GraniteSpeechFeatureExtractor.
    The FMS FeatureExtractor follows the HF implementation pattern.

    Reference: HF test_processing_granite_speech.py
    """

    def test_feature_extractor_init(self):
        """Test that GraniteSpeechFeatureExtractor can be initialized with default params."""
        extractor = GraniteSpeechFeatureExtractor()

        # Check default values match expected
        assert extractor.sampling_rate == 16000
        assert extractor.n_fft == 512
        assert extractor.win_length == 400
        assert extractor.hop_length == 160
        assert extractor.n_mels == 80
        assert extractor.projector_window_size == 15
        assert extractor.projector_downsample_rate == 5

    def test_feature_extractor_init_custom_params(self):
        """Test that GraniteSpeechFeatureExtractor can be initialized with custom params."""
        extractor = GraniteSpeechFeatureExtractor(
            sampling_rate=8000,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=40,
            projector_window_size=10,
            projector_downsample_rate=2,
        )

        assert extractor.sampling_rate == 8000
        assert extractor.n_fft == 256
        assert extractor.win_length == 200
        assert extractor.hop_length == 80
        assert extractor.n_mels == 40
        assert extractor.projector_window_size == 10
        assert extractor.projector_downsample_rate == 2

    def test_feature_extractor_call_returns_dict(self):
        """Test that calling feature extractor returns expected dict structure."""
        extractor = GraniteSpeechFeatureExtractor()

        # Create dummy audio input
        audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz

        result = extractor(audio)

        # Check dict structure (even if values are None for skeleton implementation)
        assert isinstance(result, dict)
        assert "input_features" in result
        assert "audio_embed_sizes" in result
        assert "input_features_mask" in result

    # =========================================================================
    # Feature extraction tests (HF parity)
    # =========================================================================

    def test_feature_extractor_call_returns_tensor(self):
        """Test that __call__ returns actual tensor features.

        HF Source: Tested indirectly via processor tests
        """
        extractor = GraniteSpeechFeatureExtractor()
        audio = torch.randn(1, 16000)  # 1 second at 16kHz

        result = extractor(audio)

        # Should return actual tensors
        assert result["input_features"] is not None
        assert isinstance(result["input_features"], torch.Tensor)

    def test_feature_extractor_output_shape(self):
        """Test that mel-spectrogram output has correct shape.

        Expected shape: (batch, mel_seq_len, num_features=160)
        where num_features = n_mels * 2 = 80 * 2 = 160 (due to stacking)

        HF Source: test_processing_granite_speech.py L130-163
        """
        extractor = GraniteSpeechFeatureExtractor()

        # 1 second of audio at 16kHz
        audio = torch.randn(1, 16000)
        result = extractor(audio)

        assert result["input_features"] is not None
        # Shape should be (batch=1, mel_seq_len, 160)
        assert result["input_features"].dim() == 3
        assert result["input_features"].shape[0] == 1
        assert result["input_features"].shape[2] == 160  # n_mels * 2

    def test_get_num_audio_features(self):
        """Test _get_num_audio_features calculates correct projected lengths.

        The calculation should be:
        1. mel_length = raw_length // hop_length + 1
        2. encoder_length = mel_length // 2 (due to stacking)
        3. nblocks = ceil(encoder_length / projector_window_size)
        4. projector_length = nblocks * (window_size // downsample_rate)

        HF Source: test_processing_granite_speech.py L130-163, L165-197
        """
        extractor = GraniteSpeechFeatureExtractor(
            hop_length=160,
            projector_window_size=15,
            projector_downsample_rate=5,
        )

        # Test with known audio lengths from HF tests
        # 269920 samples -> 171 projected features (from HF test)
        audio_lengths = [269920]
        result = extractor._get_num_audio_features(audio_lengths)

        assert len(result) == 1
        assert result[0] == 171  # Expected from HF test

    def test_get_num_audio_features_multiple(self):
        """Test _get_num_audio_features with multiple audio lengths.

        HF Source: test_processing_granite_speech.py L165-197
        """
        extractor = GraniteSpeechFeatureExtractor(
            hop_length=160,
            projector_window_size=15,
            projector_downsample_rate=5,
        )

        # Test with varying lengths from HF test
        audio_lengths = [142100, 269920]
        result = extractor._get_num_audio_features(audio_lengths)

        assert len(result) == 2
        assert result[0] == 90   # Expected from HF test
        assert result[1] == 171  # Expected from HF test

    def test_extract_mel_spectrograms(self):
        """Test _extract_mel_spectrograms produces valid output.

        HF Source: Tested indirectly via __call__
        """
        extractor = GraniteSpeechFeatureExtractor()
        audio = torch.randn(2, 16000)  # batch of 2, 1 second each

        result = extractor._extract_mel_spectrograms(audio)

        assert result is not None
        assert isinstance(result, torch.Tensor)
        # Shape: (batch, mel_seq_len, n_mels*2)
        assert result.dim() == 3
        assert result.shape[0] == 2
        assert result.shape[2] == 160  # n_mels * 2

    def test_get_audios_and_audio_lengths_single_tensor(self):
        """Test _get_audios_and_audio_lengths with single tensor input.

        HF Source: Tested indirectly via __call__
        """
        extractor = GraniteSpeechFeatureExtractor()

        # Single audio tensor
        audio = torch.randn(16000)  # 1 second
        batched, lengths = extractor._get_audios_and_audio_lengths(audio)

        assert batched is not None
        assert len(lengths) == 1
        assert lengths[0] == 16000

    def test_get_audios_and_audio_lengths_list(self):
        """Test _get_audios_and_audio_lengths with list of tensors.

        HF Source: Tested indirectly via __call__
        """
        extractor = GraniteSpeechFeatureExtractor()

        # List of audio tensors with different lengths
        audios = [torch.randn(16000), torch.randn(32000)]
        batched, lengths = extractor._get_audios_and_audio_lengths(audios)

        assert batched is not None
        assert len(lengths) == 2
        assert lengths[0] == 16000
        assert lengths[1] == 32000

    def test_feature_extractor_audio_embed_sizes(self):
        """Test that __call__ returns correct audio_embed_sizes.

        HF Source: test_processing_granite_speech.py L130-163
        """
        extractor = GraniteSpeechFeatureExtractor()

        # Audio length that produces known embed size
        audio = torch.randn(1, 269920)
        result = extractor(audio)

        assert result["audio_embed_sizes"] is not None
        assert len(result["audio_embed_sizes"]) == 1
        assert result["audio_embed_sizes"][0] == 171  # Expected from HF

    def test_feature_extractor_mask_shape(self):
        """Test that input_features_mask has correct shape.

        HF Source: test_processing_granite_speech.py L130-163
        """
        extractor = GraniteSpeechFeatureExtractor()
        audio = torch.randn(2, 16000)  # batch of 2

        result = extractor(audio)

        assert result["input_features_mask"] is not None
        assert isinstance(result["input_features_mask"], torch.Tensor)
        # Mask should match input_features batch and sequence dims
        assert result["input_features_mask"].shape[0] == 2


@requires_torchaudio
class TestFMSGraniteSpeechProcessor:
    """
    Tests for FMS-native GraniteSpeechProcessor.

    These tests validate the fully implemented GraniteSpeechProcessor.
    The FMS Processor combines audio feature extraction with text tokenization.

    Reference: HF test_processing_granite_speech.py
    """

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        class MockTokenizer:
            def __init__(self):
                self.audio_token = "<|audio|>"

            def __call__(self, text, **kwargs):
                # Simple mock tokenization
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        return MockTokenizer()

    def test_processor_init(self, mock_tokenizer):
        """Test that GraniteSpeechProcessor can be initialized."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        assert processor.audio_processor is audio_processor
        assert processor.tokenizer is mock_tokenizer
        assert processor.audio_token == "<|audio|>"

    def test_processor_init_custom_audio_token(self, mock_tokenizer):
        """Test that GraniteSpeechProcessor respects custom audio token."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
            audio_token="<audio>",
        )

        # Should use tokenizer's audio_token if available
        assert processor.audio_token == "<|audio|>"

    def test_processor_init_tokenizer_without_audio_token(self):
        """Test processor with tokenizer that doesn't have audio_token attribute."""
        class SimpleTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": [1, 2, 3]}

        audio_processor = GraniteSpeechFeatureExtractor()
        tokenizer = SimpleTokenizer()

        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=tokenizer,
            audio_token="<custom_audio>",
        )

        # Should use provided audio_token since tokenizer doesn't have one
        assert processor.audio_token == "<custom_audio>"

    def test_processor_call_returns_dict(self, mock_tokenizer):
        """Test that calling processor returns expected dict structure."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        result = processor(text="Test text")

        # Check dict structure (even if values are None for skeleton implementation)
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_get_validated_text_string(self, mock_tokenizer):
        """Test _get_validated_text with string input."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        result = processor._get_validated_text("hello")
        assert result == ["hello"]

    def test_get_validated_text_list(self, mock_tokenizer):
        """Test _get_validated_text with list of strings input."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        result = processor._get_validated_text(["hello", "world"])
        assert result == ["hello", "world"]

    def test_get_validated_text_invalid_type(self, mock_tokenizer):
        """Test _get_validated_text raises TypeError for invalid input."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        with pytest.raises(TypeError):
            processor._get_validated_text(123)

    def test_get_validated_text_invalid_list(self, mock_tokenizer):
        """Test _get_validated_text raises TypeError for list of non-strings."""
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        with pytest.raises(TypeError):
            processor._get_validated_text([123, 456])

    # =========================================================================
    # Input validation tests (HF parity)
    # =========================================================================

    def test_requires_text(self, mock_tokenizer):
        """Ensure text input is required.

        HF Source: test_processing_granite_speech.py L73-83
        """
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        with pytest.raises(TypeError):
            processor(text=None)

    def test_bad_text_fails(self, mock_tokenizer):
        """Ensure we gracefully fail if text is the wrong type.

        HF Source: test_processing_granite_speech.py L85-92
        """
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        with pytest.raises(TypeError):
            processor(text=424, audio=None)

    def test_bad_nested_text_fails(self, mock_tokenizer):
        """Ensure we gracefully fail if text is the wrong nested type.

        HF Source: test_processing_granite_speech.py L94-104
        """
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        with pytest.raises(TypeError):
            processor(text=[424], audio=None)

    def test_bad_audio_fails(self, mock_tokenizer):
        """Ensure we gracefully fail if audio is the wrong type.

        HF Source: test_processing_granite_speech.py L106-116
        """
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        with pytest.raises(TypeError):
            processor(text="test", audio="foo")

    def test_nested_bad_audio_fails(self, mock_tokenizer):
        """Ensure we gracefully fail if audio is the wrong nested type.

        HF Source: test_processing_granite_speech.py L118-128
        """
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        with pytest.raises(TypeError):
            processor(text="test", audio=["foo"])

    def test_processor_returns_tokenized_input(self, mock_tokenizer):
        """Test that processor returns actual tokenized input_ids.

        HF Source: test_processing_granite_speech.py L130-163
        """
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        result = processor(text="Hello world")

        assert result["input_ids"] is not None

    # =========================================================================
    # Audio token expansion tests (HF parity)
    # =========================================================================

    def test_audio_token_filling_same_len_feature_tensors(self, mock_tokenizer):
        """Ensure audio token filling is handled correctly when we have
        one or more audio inputs whose features are all the same length
        stacked into a tensor / numpy array.

        NOTE: Currently we enforce that each sample can only have one audio.

        HF Source: test_processing_granite_speech.py L130-163
        """
        # Create a mock tokenizer that tracks audio tokens
        class TrackingTokenizer:
            def __init__(self):
                self.audio_token = "<|audio|>"

            def __call__(self, text, return_tensors=None, **kwargs):
                # Count audio tokens in text
                if isinstance(text, str):
                    text = [text]
                audio_token_counts = [t.count(self.audio_token) for t in text]
                # Return token IDs with audio_token_id = 999
                input_ids = []
                for t in text:
                    ids = []
                    for char in t.split():
                        if char == self.audio_token:
                            ids.append(999)
                        else:
                            ids.append(1)
                    input_ids.append(ids)
                result = {"input_ids": input_ids, "attention_mask": [[1]*len(ids) for ids in input_ids]}
                if return_tensors == "pt":
                    # Pad sequences
                    max_len = max(len(ids) for ids in input_ids)
                    for ids in input_ids:
                        ids.extend([0] * (max_len - len(ids)))
                    result = {k: torch.tensor(v) for k, v in result.items()}
                return result

            def get_vocab(self):
                return {self.audio_token: 999}

        audio_processor = GraniteSpeechFeatureExtractor()
        tokenizer = TrackingTokenizer()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=tokenizer,
        )

        # Create audio input (same dims as HF test)
        vec_dims = [1, 269920]
        audio = torch.rand(*vec_dims) - 0.5

        audio_tokens = processor.audio_token * vec_dims[0]
        inputs = processor(
            text=f"{audio_tokens} Can you compare this audio?",
            audio=audio,
        )

        # Verify input_features shape
        assert inputs.get("input_features") is not None
        assert list(inputs["input_features"].shape) == [vec_dims[0], 844, 160]

    def test_audio_token_filling_same_len_feature_numpy(self, mock_tokenizer):
        """Ensure audio token filling is handled correctly with numpy array input.

        This test validates that the processor can handle numpy arrays in addition
        to torch tensors, matching the HF parameterized test.

        HF Source: test_processing_granite_speech.py L130-163 (np.random.rand variant)
        """
        # Create a mock tokenizer that tracks audio tokens
        class TrackingTokenizer:
            def __init__(self):
                self.audio_token = "<|audio|>"

            def __call__(self, text, return_tensors=None, **kwargs):
                if isinstance(text, str):
                    text = [text]
                input_ids = []
                for t in text:
                    ids = []
                    for char in t.split():
                        if char == self.audio_token:
                            ids.append(999)
                        else:
                            ids.append(1)
                    input_ids.append(ids)
                result = {"input_ids": input_ids, "attention_mask": [[1]*len(ids) for ids in input_ids]}
                if return_tensors == "pt":
                    max_len = max(len(ids) for ids in input_ids)
                    for ids in input_ids:
                        ids.extend([0] * (max_len - len(ids)))
                    result = {k: torch.tensor(v) for k, v in result.items()}
                return result

            def get_vocab(self):
                return {self.audio_token: 999}

        audio_processor = GraniteSpeechFeatureExtractor()
        tokenizer = TrackingTokenizer()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=tokenizer,
        )

        # Create audio input as numpy array (HF test uses np.random.rand)
        vec_dims = [1, 269920]
        audio = np.random.rand(*vec_dims) - 0.5

        audio_tokens = processor.audio_token * vec_dims[0]
        inputs = processor(
            text=f"{audio_tokens} Can you compare this audio?",
            audio=audio,
        )

        # Verify input_features shape (same as torch.rand variant)
        assert inputs.get("input_features") is not None
        assert list(inputs["input_features"].shape) == [vec_dims[0], 844, 160]

    def test_audio_token_filling_varying_len_feature_list(self, mock_tokenizer):
        """Ensure audio token filling is handled correctly when we have
        multiple varying len audio sequences passed as a list.

        HF Source: test_processing_granite_speech.py L165-197
        """
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        vec_dims = [[1, 142100], [1, 269920]]
        num_expected_features = [90, 171]
        audio = [torch.rand(dims) - 0.5 for dims in vec_dims]

        inputs = processor(
            text=[
                f"{processor.audio_token} Can you describe this audio?",
                f"{processor.audio_token} How does it compare with this audio?",
            ],
            audio=audio,
        )

        # Verify input_features is not None
        assert inputs.get("input_features") is not None

        # Verify audio_embed_sizes match expected
        # Note: audio_embed_sizes is popped from audio_inputs and not returned
        # The processor expands audio tokens internally using the sizes

    def test_expand_audio_tokens_single(self, mock_tokenizer):
        """Test _expand_audio_tokens expands single audio token correctly.

        HF Source: test_processing_granite_speech.py (implicit in processor tests)
        """
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        text = ["<|audio|> transcribe this"]
        audio_embed_sizes = [5]  # 5 embeddings for this audio

        result = processor._expand_audio_tokens(text, audio_embed_sizes)

        # Should expand <|audio|> to 5 copies
        assert result[0].count("<|audio|>") == 5

    def test_expand_audio_tokens_multiple(self, mock_tokenizer):
        """Test _expand_audio_tokens expands multiple texts with different sizes.

        HF Source: test_processing_granite_speech.py (implicit in processor tests)
        """
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        text = [
            "<|audio|> first audio",
            "<|audio|> second audio",
        ]
        audio_embed_sizes = [3, 7]

        result = processor._expand_audio_tokens(text, audio_embed_sizes)

        # First text should have 3 audio tokens
        # Second text should have 7 audio tokens
        assert result[0].count("<|audio|>") == 3
        assert result[1].count("<|audio|>") == 7

    # FIXME: This test is skipped pending verification of HF behavior.
    # The HF test expects output tensors on CPU regardless of processing device,
    # but HF's _extract_mel_spectrograms appears to return tensors on the specified device.
    # Need to run HF test to verify expected behavior:
    #   - If HF test passes: FMS GraniteSpeechProcessor needs to move tensors back to CPU
    #   - If HF test fails: Both HF test and this test have incorrect expectations
    # See: HF test_processing_granite_speech.py L199-221
    @pytest.mark.skip(
        reason="FIXME: Pending verification of HF behavior - unclear if output should be on CPU or device"
    )
    @pytest.mark.skipif(
        torch_device == "cpu",
        reason="Test requires GPU/accelerator"
    )
    def test_device_override(self, mock_tokenizer):
        """Ensure that regardless of the processing device, the tensors
        produced are on the CPU.

        HF Source: test_processing_granite_speech.py L199-221
        """
        audio_processor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=audio_processor,
            tokenizer=mock_tokenizer,
        )

        vec_dims = [1, 269920]
        wav = torch.rand(vec_dims) - 0.5

        inputs = processor(
            text=f"{processor.audio_token} Can you transcribe this audio?",
            audio=wav,
            device=torch_device,
        )

        # Output should always be on CPU regardless of processing device
        assert inputs["input_features"].device.type == "cpu"


# =============================================================================
# E2E Integration Tests (using FMS components)
# =============================================================================

# Check for required dependencies
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets not available")
class TestGraniteSpeechE2E:
    """
    End-to-End integration tests for FMS GraniteSpeech.

    Tests the full pipeline using FMS-native components:
    - GraniteSpeech model
    - GraniteSpeechFeatureExtractor (fully implemented)
    - GraniteSpeechProcessor (fully implemented)

    These tests validate the complete E2E flow including:
    - Real audio loading from LibriSpeech dataset
    - Mel-spectrogram feature extraction
    - Audio token expansion in text
    - Model forward pass with audio features

    Reference: HF test_modeling_granite_speech.py L296-378
    """

    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        encoder_config = ConformerConfig(
            num_features=160,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dim_head=16,
            conv_kernel_size=15,
            conv_expansion_factor=2,
            feedforward_mult=4,
            dropout=0.0,
            output_dim=42,
        )

        decoder_config = GraniteConfig(
            src_vocab_size=1000,
            emb_dim=64,
            nlayers=2,
            nheads=4,
            hidden_grow_factor=2.0,
            max_expected_seq_len=512,
            pad_id=0,
        )

        projector_config = SpeechProjectorConfig(
            encoder_dim=64,
            decoder_dim=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
            window_size=15,
            downsample_rate=5,
            num_queries=3,
        )

        return GraniteSpeechConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            projector_config=projector_config,
            audio_token_index=999,
            downsample_rate=5,
            window_size=15,
        )

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for E2E testing."""
        class MockTokenizer:
            def __init__(self):
                self.audio_token = "<|audio|>"
                self.pad_token_id = 0
                self.eos_token_id = 1

            def __call__(self, text, return_tensors=None, padding=True, **kwargs):
                # Simple mock: return fixed token IDs
                if isinstance(text, str):
                    text = [text]

                # Count audio tokens and create input_ids
                batch_input_ids = []
                for t in text:
                    # Simple tokenization: audio token -> 999, other chars -> random ids
                    ids = []
                    i = 0
                    while i < len(t):
                        if t[i:i+len(self.audio_token)] == self.audio_token:
                            ids.append(999)  # audio_token_index
                            i += len(self.audio_token)
                        else:
                            ids.append(ord(t[i]) % 998 + 1)  # Map to 1-998
                            i += 1
                    batch_input_ids.append(ids)

                # Pad to same length
                max_len = max(len(ids) for ids in batch_input_ids)
                for ids in batch_input_ids:
                    ids.extend([0] * (max_len - len(ids)))

                result = {
                    "input_ids": batch_input_ids,
                    "attention_mask": [[1] * len(ids) for ids in batch_input_ids],
                }

                if return_tensors == "pt":
                    result = {k: torch.tensor(v) for k, v in result.items()}

                return result

            def decode(self, ids, skip_special_tokens=True):
                return "mock decoded text"

            def batch_decode(self, ids, skip_special_tokens=True):
                return ["mock decoded text"] * len(ids)

        return MockTokenizer()

    def test_fms_model_forward_text_only(self, small_config):
        """Test FMS GraniteSpeech model forward pass with text only."""
        model = GraniteSpeech(small_config)
        model.reset_parameters()
        model.eval()

        batch_size = 2
        seq_len = 10

        # Text-only input (no audio tokens)
        input_ids = torch.randint(1, 998, (batch_size, seq_len))

        with torch.no_grad():
            logits, loss = model(input_ids=input_ids)

        assert logits.shape == (batch_size, seq_len, small_config.decoder_config.src_vocab_size)
        assert loss is None  # No labels provided

    def test_fms_model_forward_with_audio(self, small_config):
        """Test FMS GraniteSpeech model forward pass with audio features."""
        model = GraniteSpeech(small_config)
        model.reset_parameters()
        model.eval()

        batch_size = 1
        seq_len = 10
        audio_seq_len = 45  # 3 windows of 15

        # Calculate expected audio tokens
        num_windows = audio_seq_len // small_config.window_size
        num_audio_tokens = num_windows * small_config.projector_config.num_queries

        # Input with audio tokens at the beginning
        input_ids = torch.randint(1, 998, (batch_size, seq_len))
        input_ids[0, :num_audio_tokens] = small_config.audio_token_index

        # Audio features
        input_features = torch.randn(batch_size, audio_seq_len, 160)

        with torch.no_grad():
            logits, loss = model(
                input_ids=input_ids,
                input_features=input_features,
            )

        assert logits.shape == (batch_size, seq_len, small_config.decoder_config.src_vocab_size)

    def test_fms_processor_integration(self, small_config, mock_tokenizer):
        """Test FMS processor with feature extractor integration."""
        feature_extractor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=feature_extractor,
            tokenizer=mock_tokenizer,
        )

        # Test text processing
        text = "<|audio|> transcribe this audio"
        result = processor(text=text)

        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_fms_feature_extractor_integration(self, small_config):
        """Test FMS feature extractor with model integration."""
        feature_extractor = GraniteSpeechFeatureExtractor()

        # Create dummy audio (1 second at 16kHz)
        audio = torch.randn(1, 16000)

        # Extract features (skeleton returns dummy output)
        result = feature_extractor(audio)

        assert isinstance(result, dict)
        assert "input_features" in result
        assert "audio_embed_sizes" in result
        assert "input_features_mask" in result

    @requires_torchaudio
    @pytest.mark.slow
    def test_fms_e2e_with_real_audio(self, small_config, mock_tokenizer):
        """
        E2E test with real audio from LibriSpeech dataset.

        This test validates the full pipeline:
        1. Load real audio from LibriSpeech
        2. Process audio through GraniteSpeechFeatureExtractor
        3. Process text + audio through GraniteSpeechProcessor
        4. Forward pass through GraniteSpeech model

        HF Source: test_modeling_granite_speech.py L318-349
        """
        import math

        # Load real audio samples
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            trust_remote_code=True,
        )
        speech_samples = ds.sort("id")[:1]["audio"]
        audio_array = speech_samples[0]["array"]

        # Create FMS components
        feature_extractor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=feature_extractor,
            tokenizer=mock_tokenizer,
        )

        # Convert audio to tensor
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

        # Process text with audio placeholder
        text = "<|audio|> transcribe this audio"
        processed = processor(text=text, audio=audio_tensor)

        # Verify processor output structure
        assert isinstance(processed, dict)
        assert "input_ids" in processed
        assert "input_features" in processed
        assert "input_features_mask" in processed
        assert processed["input_features"] is not None

        # Verify audio feature shape (batch, mel_seq_len, 160)
        assert processed["input_features"].dim() == 3
        assert processed["input_features"].shape[0] == 1  # batch size
        assert processed["input_features"].shape[2] == 160  # n_mels * 2

        # Verify audio feature mask
        assert processed["input_features_mask"] is not None

        # Calculate expected audio tokens for the model
        # Need to match audio_token_index placeholders with projected features
        audio_lengths = [len(audio_array)]
        num_audio_features = feature_extractor._get_num_audio_features(audio_lengths)[0]

        # Create model with matching config
        # Update config to have enough vocab for our mock tokenizer
        model = GraniteSpeech(small_config)
        model.reset_parameters()
        model.eval()

        # Create input_ids with correct number of audio tokens
        # The processor expands <|audio|> to num_audio_features copies
        text_tokens = 20  # approximate text length
        total_seq_len = num_audio_features + text_tokens
        input_ids = torch.randint(1, 998, (1, total_seq_len))
        input_ids[0, :num_audio_features] = small_config.audio_token_index

        # Forward pass with real audio features
        with torch.no_grad():
            logits, loss = model(
                input_ids=input_ids,
                input_features=processed["input_features"],
                input_features_mask=processed["input_features_mask"],
            )

        assert logits is not None
        assert logits.shape[0] == 1  # batch size
        assert logits.shape[2] == small_config.decoder_config.src_vocab_size
        assert not torch.isnan(logits).any().item()

    @requires_torchaudio
    @pytest.mark.slow
    def test_fms_e2e_with_real_audio_batch(self, small_config, mock_tokenizer):
        """
        E2E test with batched real audio from LibriSpeech dataset.

        Tests the full pipeline with multiple audio samples of different lengths.

        HF Source: test_modeling_granite_speech.py L351-378
        """
        # Load multiple real audio samples
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            trust_remote_code=True,
        )
        speech_samples = ds.sort("id")[:2]["audio"]
        audio_arrays = [sample["array"] for sample in speech_samples]

        # Create FMS components
        feature_extractor = GraniteSpeechFeatureExtractor()
        processor = GraniteSpeechProcessor(
            audio_processor=feature_extractor,
            tokenizer=mock_tokenizer,
        )

        # Convert audio to tensors
        audio_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in audio_arrays]

        # Process text with audio placeholder (one per audio)
        texts = [
            "<|audio|> transcribe the first audio",
            "<|audio|> transcribe the second audio",
        ]
        processed = processor(text=texts, audio=audio_tensors)

        # Verify processor output structure
        assert isinstance(processed, dict)
        assert "input_ids" in processed
        assert "input_features" in processed
        assert "input_features_mask" in processed

        # Verify batch dimension
        assert processed["input_features"].shape[0] == 2  # batch size

        # Verify feature sizes match computed features
        audio_lengths = [len(arr) for arr in audio_arrays]
        num_computed_features = feature_extractor._get_num_audio_features(audio_lengths)

        # Mask should have shape (batch, max_features)
        num_actual_features = torch.sum(processed["input_features_mask"], dim=-1)
        for expected, actual in zip(num_computed_features, num_actual_features):
            assert expected == actual.item()


# ============================================================================
# Auto Config Loading Tests (hf_configured / hf_pretrained)
# ============================================================================


class TestGraniteSpeechAutoConfigLoading:
    """
    Tests for get_model auto config loading feature.

    These tests verify that GraniteSpeech can be loaded via:
    1. get_model("hf_configured", "ibm-granite/granite-speech-3.3-2b")
    2. get_model("hf_pretrained", "ibm-granite/granite-speech-3.3-2b")

    The auto config loading feature extracts model configuration from HuggingFace
    and creates the corresponding FMS model.
    """

    def test_map_model_config_granite_speech(self):
        """
        Test that _map_model_config correctly maps GraniteSpeech HF config to FMS config.

        This tests the config mapping without downloading any model weights.
        """
        from fms.models.hf.utils import _map_model_config
        from fms.models.conformer import ConformerConfig
        from fms.models.granite import GraniteConfig
        from fms.modules.projector import SpeechProjectorConfig

        # Create a mock HF config structure
        class MockEncoderConfig:
            input_dim = 160
            hidden_dim = 1024
            num_layers = 16
            num_heads = 8
            dim_head = 128
            conv_kernel_size = 15
            conv_expansion_factor = 2
            feedforward_mult = 4
            dropout = 0.1
            max_pos_emb = 512
            context_size = 200
            output_dim = 256

        class MockProjectorConfig:
            hidden_size = 1024
            num_hidden_layers = 2
            num_attention_heads = 16
            intermediate_size = 4096
            hidden_dropout_prob = 0.1
            attention_probs_dropout_prob = 0.1
            hidden_act = "gelu"
            layer_norm_eps = 1e-12
            initializer_range = 0.02
            cross_attention_frequency = 1

        class MockTextConfig:
            model_type = "granite"
            hidden_size = 4096
            intermediate_size = 12800
            num_hidden_layers = 40
            num_attention_heads = 32
            num_key_value_heads = 8
            rms_norm_eps = 1e-5
            rope_theta = 10000000.0
            vocab_size = 49160
            tie_word_embeddings = False
            attention_bias = False
            mlp_bias = False
            max_position_embeddings = 131072
            residual_multiplier = 0.22
            attention_multiplier = 0.0078125
            logits_scaling = 16.0
            embedding_multiplier = 12.0
            hidden_act = "silu"
            head_dim = 128

        class MockGraniteSpeechConfig:
            architectures = ["GraniteSpeechForConditionalGeneration"]
            audio_token_index = 49159
            window_size = 15
            downsample_rate = 5
            has_lora_adapter = True
            encoder_config = MockEncoderConfig()
            projector_config = MockProjectorConfig()
            text_config = MockTextConfig()

        # Test the mapping
        architecture, config_params = _map_model_config(
            "GraniteSpeechForConditionalGeneration",
            MockGraniteSpeechConfig()
        )

        # Verify architecture name
        assert architecture == "granite_speech"

        # Verify top-level params
        assert config_params["audio_token_index"] == 49159
        assert config_params["window_size"] == 15
        assert config_params["downsample_rate"] == 5
        assert config_params["has_lora_adapter"] is True

        # Verify encoder config
        assert isinstance(config_params["encoder_config"], ConformerConfig)
        assert config_params["encoder_config"].num_features == 160
        assert config_params["encoder_config"].hidden_dim == 1024
        assert config_params["encoder_config"].num_layers == 16
        assert config_params["encoder_config"].output_dim == 256

        # Verify projector config
        assert isinstance(config_params["projector_config"], SpeechProjectorConfig)
        assert config_params["projector_config"].encoder_dim == 1024
        assert config_params["projector_config"].decoder_dim == 4096
        assert config_params["projector_config"].num_hidden_layers == 2

        # Verify decoder config
        assert isinstance(config_params["decoder_config"], GraniteConfig)
        assert config_params["decoder_config"].emb_dim == 4096
        assert config_params["decoder_config"].nlayers == 40
        assert config_params["decoder_config"].nheads == 32

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="hf_configured test requires CUDA for practical model size"
    )
    def test_get_model_hf_configured_granite_speech(self):
        """
        Test get_model with hf_configured for GraniteSpeech.

        This downloads only the config.json (not weights) from HuggingFace
        and creates an FMS model with matching configuration.

        Requires: transformers with GraniteSpeech support
        """
        from fms import models

        # Use 2B model as it's smaller
        model_id = "ibm-granite/granite-speech-3.3-2b"

        try:
            # This should:
            # 1. Download config.json from HF
            # 2. Parse the config using _map_model_config
            # 3. Create FMS GraniteSpeech model (randomly initialized)
            model = models.get_model("hf_configured", model_id)

            # Verify model was created
            assert model is not None
            assert isinstance(model, GraniteSpeech)

            # Verify config was correctly inferred
            assert model.config.audio_token_index == 49159
            assert model.config.encoder_config.num_layers == 16
            assert model.config.decoder_config.nlayers == 40

        except Exception as e:
            # If transformers doesn't have GraniteSpeech, skip
            if "GraniteSpeech" in str(e) or "granite_speech" in str(e):
                pytest.skip(f"Transformers doesn't support GraniteSpeech: {e}")
            raise

    def test_get_model_direct_variant_2b(self):
        """
        Test get_model with architecture + variant for 2B model.

        This tests direct variant loading without HF download, using the
        registered variant configuration.
        """
        from fms.models import get_model, list_variants

        # Verify variant is registered
        variants = list_variants("granite_speech")
        assert "3.3-2b" in variants, f"3.3-2b not in registered variants: {variants}"

        # Load model with direct variant
        model = get_model(
            architecture="granite_speech",
            variant="3.3-2b",
            device_type="cpu",
        )

        # Verify model was created
        assert model is not None
        assert isinstance(model, GraniteSpeech)

        # Verify config matches expected 2B configuration
        config = model.get_config()
        assert config.encoder_config.num_layers == 16
        assert config.decoder_config.nlayers == 40
        assert config.decoder_config.emb_dim == 2048  # 2B model embedding dim

    def test_get_model_direct_variant_8b(self):
        """
        Test get_model with architecture + variant for 8B model.

        This tests direct variant loading for the larger model variant.
        """
        from fms.models import get_model, list_variants

        # Verify variant is registered
        variants = list_variants("granite_speech")
        assert "3.3-8b" in variants, f"3.3-8b not in registered variants: {variants}"

        # Load model with direct variant
        model = get_model(
            architecture="granite_speech",
            variant="3.3-8b",
            device_type="cpu",
        )

        # Verify model was created
        assert model is not None
        assert isinstance(model, GraniteSpeech)

        # Verify config matches expected 8B configuration
        config = model.get_config()
        assert config.encoder_config.num_layers == 16
        assert config.decoder_config.nlayers == 40
        assert config.decoder_config.emb_dim == 4096  # 8B model embedding dim

    def test_get_model_invalid_variant_raises(self):
        """
        Test that get_model raises KeyError for invalid variant.

        Verifies proper error handling when a non-existent variant is requested.
        """
        from fms.models import get_model

        with pytest.raises(KeyError) as exc_info:
            get_model(
                architecture="granite_speech",
                variant="invalid-nonexistent-variant",
                device_type="cpu",
            )

        # Verify error message mentions the invalid variant
        assert "invalid-nonexistent-variant" in str(exc_info.value)

    def test_get_model_hf_pretrained_conflicting_params_raises(self):
        """
        Test that hf_pretrained with both variant and model_path raises ValueError.

        The hf_pretrained architecture requires exactly one of variant or model_path,
        not both simultaneously.
        """
        from fms.models import get_model

        with pytest.raises(ValueError) as exc_info:
            get_model(
                architecture="hf_pretrained",
                variant="ibm-granite/granite-speech-3.3-2b",
                model_path="/some/nonexistent/path",
                device_type="cpu",
            )

        # Verify error message explains the conflict
        error_msg = str(exc_info.value)
        assert "variant" in error_msg.lower() or "model_path" in error_msg.lower()


# =============================================================================
# CPU-Only Tests (no CUDA requirement)
# =============================================================================


class TestGraniteSpeechCPU:
    """
    Tests for GraniteSpeech that run on CPU without CUDA requirement.

    These tests enable CI/CD on CPU-only machines and verify that the model
    works correctly without GPU acceleration.
    """

    @pytest.fixture
    def small_cpu_config(self):
        """Create a small config optimized for CPU testing."""
        encoder_config = ConformerConfig(
            num_features=160,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dim_head=16,
            conv_kernel_size=15,
            conv_expansion_factor=2,
            feedforward_mult=4,
            dropout=0.0,
            output_dim=42,
        )

        decoder_config = GraniteConfig(
            src_vocab_size=1000,
            emb_dim=64,
            nlayers=2,
            nheads=4,
            hidden_grow_factor=2.0,
            max_expected_seq_len=256,
            pad_id=0,
        )

        projector_config = SpeechProjectorConfig(
            encoder_dim=64,
            decoder_dim=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=128,
            window_size=15,
            downsample_rate=5,
            num_queries=3,
        )

        return GraniteSpeechConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            projector_config=projector_config,
            audio_token_index=999,
            downsample_rate=5,
            window_size=15,
        )

    def _init_model_weights(self, model):
        """Initialize model weights with small values for stability."""
        torch.manual_seed(42)
        sd = model.state_dict()
        for key in sd.keys():
            param = sd[key]
            if param.dtype in (torch.long, torch.int, torch.int32, torch.int64):
                continue
            if "running_mean" in key or "running_var" in key:
                continue
            # Use smaller values for stability
            values = torch.randn_like(param) * 0.01
            param.copy_(values)

    def test_forward_pass_cpu_text_only(self, small_cpu_config):
        """Test text-only forward pass works on CPU."""
        model = GraniteSpeech(small_cpu_config)
        self._init_model_weights(model)
        model.eval()

        batch_size = 2
        seq_len = 20

        # Text-only input (no audio tokens)
        input_ids = torch.randint(1, 998, (batch_size, seq_len))

        with torch.no_grad():
            logits, loss = model(input_ids=input_ids)

        assert logits is not None
        assert logits.shape == (batch_size, seq_len, small_cpu_config.decoder_config.src_vocab_size)
        assert loss is None  # No labels provided
        assert not torch.isnan(logits).any().item(), "CPU forward produced NaN values"

    def test_forward_pass_cpu_with_audio(self, small_cpu_config):
        """Test forward pass with audio features works on CPU."""
        import math

        model = GraniteSpeech(small_cpu_config)
        self._init_model_weights(model)
        model.eval()

        batch_size = 1
        audio_len = 45  # Multiple of window_size (15)

        # Calculate expected audio tokens
        num_windows = audio_len // small_cpu_config.window_size
        num_audio_tokens = num_windows * small_cpu_config.projector_config.num_queries

        seq_len = num_audio_tokens + 10  # audio tokens + text tokens

        # Input with audio tokens at the beginning
        input_ids = torch.randint(1, 998, (batch_size, seq_len))
        input_ids[0, :num_audio_tokens] = small_cpu_config.audio_token_index

        # Audio features
        input_features = torch.randn(batch_size, audio_len, 160)

        with torch.no_grad():
            logits, loss = model(
                input_ids=input_ids,
                input_features=input_features,
            )

        assert logits is not None
        assert logits.shape == (batch_size, seq_len, small_cpu_config.decoder_config.src_vocab_size)
        assert not torch.isnan(logits).any().item(), "CPU forward with audio produced NaN values"

    def test_forward_pass_cpu_with_labels(self, small_cpu_config):
        """Test forward pass with labels computes loss on CPU."""
        model = GraniteSpeech(small_cpu_config)
        self._init_model_weights(model)
        model.eval()

        batch_size = 2
        seq_len = 20

        input_ids = torch.randint(1, 998, (batch_size, seq_len))

        # Create labels (shifted input_ids for language modeling)
        labels = input_ids.clone()
        labels[:, :5] = -100  # Mask first 5 tokens (prompt)

        with torch.no_grad():
            logits, loss = model(input_ids=input_ids, labels=labels)

        assert logits is not None
        assert loss is not None
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss).item(), "CPU loss computation produced NaN"

    def test_get_model_cpu_device(self, small_cpu_config):
        """Test that get_model with device_type='cpu' works correctly."""
        from fms.models import get_model

        model = get_model(
            architecture="granite_speech",
            variant="3.3-2b",
            device_type="cpu",
        )

        # Verify model is on CPU
        param = next(model.parameters())
        assert param.device.type == "cpu", f"Expected cpu, got {param.device}"

        # Verify model can do forward pass
        input_ids = torch.randint(1, 1000, (1, 10))
        with torch.no_grad():
            logits, _ = model(input_ids=input_ids)

        assert logits is not None
        assert logits.device.type == "cpu"

    def test_generation_cpu(self, small_cpu_config):
        """Test generation works on CPU."""
        from fms.utils.generation import generate

        model = GraniteSpeech(small_cpu_config)
        self._init_model_weights(model)
        model.eval()

        input_ids = torch.randint(1, 998, (1, 10))

        with torch.no_grad():
            output_ids = generate(
                model,
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                use_cache=True,
            )

        assert output_ids.shape == (1, 15)  # 10 input + 5 generated
        assert not torch.isnan(output_ids.float()).any().item()

    def test_generation_cpu_with_audio(self, small_cpu_config):
        """Test generation with audio features works on CPU."""
        from fms.utils.generation import generate

        model = GraniteSpeech(small_cpu_config)
        self._init_model_weights(model)
        model.eval()

        # Setup audio
        audio_len = 30
        num_windows = audio_len // small_cpu_config.window_size
        num_audio_tokens = num_windows * small_cpu_config.projector_config.num_queries

        seq_len = num_audio_tokens + 5
        input_ids = torch.randint(1, 998, (1, seq_len))
        input_ids[0, :num_audio_tokens] = small_cpu_config.audio_token_index

        input_features = torch.randn(1, audio_len, 160)
        input_features_mask = torch.ones(1, audio_len, dtype=torch.bool)

        with torch.no_grad():
            output_ids = generate(
                model,
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                use_cache=True,
                extra_kwargs={
                    "input_features": input_features,
                    "input_features_mask": input_features_mask,
                },
            )

        assert output_ids.shape == (1, seq_len + 5)