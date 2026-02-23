import pytest
import torch

from fms.models.qwen3 import Qwen3, Qwen3Config, Qwen3Headless
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class Qwen3Fixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Qwen3 Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: Qwen3Config):
        return Qwen3(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return Qwen3Config(
            src_vocab_size=384,
            emb_dim=64,
            norm_eps=1e-05,
            nheads=32,
            head_dim=64 // 32,
            kvheads=16,
            nlayers=2,
            hidden_grow_factor=2.0,
            max_expected_seq_len=1024,
        )

    @pytest.fixture(scope="class", autouse=True)
    def model(self, uninitialized_model: torch.nn.Module):
        """include this fixture to get a model that is fully initialized"""

        torch.random.manual_seed(5)
        sd = uninitialized_model.state_dict()
        params = sorted(sd.keys())
        for key in params:
            parameter = sd[key]
            opt_parameter_initialized = self._maybe_get_initialized_parameter(
                key, parameter
            )
            if opt_parameter_initialized is not None:
                parameter.copy_(opt_parameter_initialized)
            else:
                values = torch.randn_like(parameter)
                values -= 0.5
                values /= 20.0
                parameter.copy_(values)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = uninitialized_model.to(device)

        # Pre-compute RoPE frequencies for the target device to avoid graph breaks
        model.base_model.rot_emb.compute_freqs_cis(
            torch.device(device), model.base_model.config.max_expected_seq_len
        )

        return model


class TestQwen3(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    Qwen3Fixtures,
):
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_model_compile_no_graph_breaks(self, model):
        """Test that Qwen3 model is compilable without graph breaks"""
        import platform
        from torch._dynamo.exc import TorchDynamoException
        from torch._dynamo.testing import CompileCounterWithBackend
        from fms.testing.comparison import get_signature

        if platform.system() != "Linux":
            pytest.skip(
                f"pytorch compile is more stable on Linux, skipping as current platform is {platform.platform()}"
            )

        try:
            torch._dynamo.reset()

            # Move model to the appropriate device before compilation
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
            # default attn_algorithm won't compile on CPU for older pytorch versions
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
            pytest.fail(f"Failed to get signature of full-graph compiled model:\n{e}")

    def test_model_unfused(self, model, signature):
        pytest.skip("weight unfuse is not implemented")
