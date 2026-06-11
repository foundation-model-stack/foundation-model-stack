import pytest
import torch

from fms.models.qwen3 import Qwen3, Qwen3Config
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class Qwen3DecoderFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Qwen3 decoder fixtures mirroring the _4b_config structure:
    - 4:1 GQA ratio (nheads=8, kvheads=2, like 4B's 32:8)
    - explicit head_dim decoupled from emb_dim/nheads (like 4B's head_dim=128)
    - larger hidden_grow_factor (~3.8, same ratio as 4B's 9728/2560)
    - tie_heads=True (matching the 4B config)
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: Qwen3Config):
        return Qwen3(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return Qwen3Config(
            src_vocab_size=512,
            emb_dim=256,
            norm_eps=1e-6,
            nheads=8,
            kvheads=2,
            nlayers=2,
            head_dim=32,
            hidden_grow_factor=9728 / 2560,
            max_expected_seq_len=1024,
            rope_theta=1000000.0,
            tie_heads=True,
        )

    @pytest.fixture(scope="class", autouse=True)
    def model(self, uninitialized_model: torch.nn.Module):
        """Fully initialized model on the target device with RoPE pre-computed."""
        torch.random.manual_seed(42)
        sd = uninitialized_model.state_dict()
        for key in sorted(sd.keys()):
            parameter = sd[key]
            opt = self._maybe_get_initialized_parameter(key, parameter)
            if opt is not None:
                parameter.copy_(opt)
            else:
                values = torch.randn_like(parameter)
                values -= 0.5
                values /= 20.0
                parameter.copy_(values)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = uninitialized_model.to(device)
        model.base_model.rot_emb.compute_freqs_cis(
            torch.device(device), model.base_model.config.max_expected_seq_len
        )
        return model


class TestQwen3Decoder(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    Qwen3DecoderFixtures,
):
    """
    Test suite for the Qwen3 decoder model (4B-style configuration).

    Covers:
    - model configuration round-trips
    - output consistency against stored signatures
    - torch.compile fullgraph compilation
    - weight key consistency
    """

    _get_signature_params = ["x"]

    def test_config_passed_to_model_and_updated(self, model, config):
        """Model constructor merges kwargs into config without mutating the original."""
        new_model = type(model)(config=config, pad_id=config.pad_id + 1)
        assert new_model.get_config() is not config
        config.pad_id = config.pad_id + 1
        assert new_model.get_config().as_dict() == config.as_dict()

    def test_model_unfused(self, model, signature):
        pytest.skip("weight unfuse is not implemented for Qwen3")

    def test_model_compile_no_graph_breaks(self, model):
        """Qwen3 decoder is compilable without graph breaks."""
        import platform
        from torch._dynamo.exc import TorchDynamoException
        from torch._dynamo.testing import CompileCounterWithBackend
        from fms.testing.comparison import get_signature

        if platform.system() != "Linux":
            pytest.skip(
                f"pytorch compile is more stable on Linux, skipping on {platform.platform()}"
            )

        try:
            torch._dynamo.reset()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            cnt = CompileCounterWithBackend("inductor")
            compiled_model = torch.compile(model=model, backend=cnt, fullgraph=True)
            assert cnt.frame_count == 0

            optional_params = (
                self._get_signature_optional_params.copy()
                if self._get_signature_optional_params
                else {}
            )
            optional_params["attn_algorithm"] = "math"

            get_signature(
                compiled_model,
                params=self._get_signature_params,
                optional_params=optional_params,
                logits_getter_fn=self._get_signature_logits_getter_fn,
                device=device,
            )
            assert cnt.frame_count == 1
        except TorchDynamoException as e:
            pytest.fail(f"Failed to compile Qwen3 decoder with fullgraph:\n{e}")


class Qwen3DecoderNoTieHeadsFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Fixtures for a Qwen3 decoder with tie_heads=False.
    Verifies that the model works correctly when the LM head and embedding
    weights are not tied (e.g., when loading non-tied checkpoints).
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: Qwen3Config):
        return Qwen3(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return Qwen3Config(
            src_vocab_size=512,
            emb_dim=256,
            norm_eps=1e-6,
            nheads=8,
            kvheads=2,
            nlayers=2,
            head_dim=32,
            hidden_grow_factor=9728 / 2560,
            max_expected_seq_len=1024,
            rope_theta=1000000.0,
            tie_heads=False,
        )

    @pytest.fixture(scope="class", autouse=True)
    def model(self, uninitialized_model: torch.nn.Module):
        torch.random.manual_seed(7)
        sd = uninitialized_model.state_dict()
        for key in sorted(sd.keys()):
            parameter = sd[key]
            opt = self._maybe_get_initialized_parameter(key, parameter)
            if opt is not None:
                parameter.copy_(opt)
            else:
                values = torch.randn_like(parameter)
                values -= 0.5
                values /= 20.0
                parameter.copy_(values)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = uninitialized_model.to(device)
        model.base_model.rot_emb.compute_freqs_cis(
            torch.device(device), model.base_model.config.max_expected_seq_len
        )
        return model


class TestQwen3DecoderNoTieHeads(
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    Qwen3DecoderNoTieHeadsFixtures,
):
    """Test Qwen3 decoder with tie_heads=False (independent embedding and LM head weights)."""

    _get_signature_params = ["x"]

    def test_model_unfused(self, model, signature):
        pytest.skip("weight unfuse is not implemented for Qwen3")

    def test_model_compile_no_graph_breaks(self, model):
        """Qwen3 decoder with untied heads is compilable without graph breaks."""
        import platform
        from torch._dynamo.exc import TorchDynamoException
        from torch._dynamo.testing import CompileCounterWithBackend
        from fms.testing.comparison import get_signature

        if platform.system() != "Linux":
            pytest.skip(
                f"pytorch compile is more stable on Linux, skipping on {platform.platform()}"
            )

        try:
            torch._dynamo.reset()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            cnt = CompileCounterWithBackend("inductor")
            compiled_model = torch.compile(model=model, backend=cnt, fullgraph=True)
            assert cnt.frame_count == 0

            optional_params = (
                self._get_signature_optional_params.copy()
                if self._get_signature_optional_params
                else {}
            )
            optional_params["attn_algorithm"] = "math"

            get_signature(
                compiled_model,
                params=self._get_signature_params,
                optional_params=optional_params,
                logits_getter_fn=self._get_signature_logits_getter_fn,
                device=device,
            )
            assert cnt.frame_count == 1
        except TorchDynamoException as e:
            pytest.fail(
                f"Failed to compile Qwen3 decoder (no tie_heads) with fullgraph:\n{e}"
            )
