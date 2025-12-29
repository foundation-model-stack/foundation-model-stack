import pytest

from fms.models.gpt_oss import GptOss, GptOssConfig
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig

import torch

from torch._dynamo.exc import TorchDynamoException
from torch._dynamo.testing import CompileCounterWithBackend

from fms.testing.comparison import get_signature


class GptOssFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base GptOss Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: GptOssConfig):
        model = GptOss(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.base_model.post_init()
        return model

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        gpt_oss_config = GptOssConfig(
            src_vocab_size=384,
            sliding_window=4,
            head_dim=16,
            norm_eps=1e-05,
            nheads=4,
            kvheads=1,
            nlayers=2,
            num_experts=8,
            rope_base=150000.0,
            rope_scaling_factor=32.0,
            rope_ntk_alpha=1.0,
            rope_ntk_beta=32.0,
        )
        return gpt_oss_config


class TestGptOss(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    GptOssFixtures,
):
    """
    Model Test Suite for GptOss

    This suite will include tests for:
    - model configuration
    - basic load/save model
    - consistency of model output
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_model_compile_no_graph_breaks(self, model):
        """Test that an FMS model is compilable without graph breaks"""
        try:
            torch._dynamo.reset()
            cnt = CompileCounterWithBackend("inductor")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            compiled_model = torch.compile(model=model, backend=cnt, fullgraph=True)
            assert cnt.frame_count == 0

            optional_params = self._get_signature_optional_params
            # default attn_algorithm won't compile on CPU for older pytorch versions
            # TODO: add non-math attn_algorithm when we have GPUs to run unit tests
            optional_params.update({"attn_algorithm": "math"})

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

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(config=config, nlayers=config.nlayers + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.nlayers = config.nlayers + 1
        assert model.get_config().as_dict() == config.as_dict()

    def test_model_unfused(self, model, signature):
        pytest.skip("weight unfuse is not implemented")
