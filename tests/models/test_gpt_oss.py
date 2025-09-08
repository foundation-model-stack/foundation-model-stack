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


class GptOssFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base GptOss Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: GptOssConfig):
        model = GptOss(config)
        model.reset_parameters()
        return model

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return GptOssConfig(
            num_experts=128,
            emb_dim=2880,
            head_dim=64,
            num_attention_heads=64,
            sliding_window=128,
            rope_base=150000.0,
            tie_heads=False,
            activation_fn= "silu",
            max_expected_seq_len=131072,
            top_k_experts=4,
            output_router_logits=False,
            layer_types=None,
            pad_id=-1,
            nheads=64,
            nlayers=24,
            kvheads=8,
        )


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

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(config=config, nlayers=config.nlayers + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.nlayers = config.nlayers + 1
        assert model.get_config().as_dict() == config.as_dict()
