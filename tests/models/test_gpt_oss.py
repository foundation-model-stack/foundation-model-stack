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
        gpt_oss_config = GptOssConfig(
            src_vocab_size=201088,
            num_experts=32,
            emb_dim=2880,
            head_dim=64,
            sliding_window=128,
            nheads=64,
            nlayers=24,
            kvheads=8,
        )
        gpt_oss_config.layer_types = [
            "sliding_attention" if bool((i + 1) % 2) else "full_attention"
            for i in range(24)
        ]
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
