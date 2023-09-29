import pytest

from fms.models.llama import LLaMA, LLaMAConfig
from fms.testing._internal.model_test_suite import (
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ConfigFixtureMixin,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class LLaMA2Fixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base LLaMA 2 Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: LLaMAConfig):
        return LLaMA(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return LLaMAConfig(
            src_vocab_size=381,
            emb_dim=16,
            norm_eps=1e-05,
            nheads=2,
            kvheads=0,
            nlayers=2,
            pad_id=0,
            hidden_grow_factor=2.6666666666666665,
            multiple_of=2,
            activation_fn="swish",
            p_dropout=0.0,
            max_expected_seq_len=4096,
            ntk_scaling=False,
        )


class TestLlama2(ModelConfigTestSuite, ModelConsistencyTestSuite, LLaMA2Fixtures):
    """
    Model Test Suite for llama

    This suite will include tests for:
    - model configuration
    - basic load/save model
    - consistency of model output
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(config=config, pad_id=config.pad_id + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.pad_id = config.pad_id + 1
        assert model.get_config().as_dict() == config.as_dict()


class LLaMA2GQAFixtures(ModelFixtureMixin):
    """
    Base LLaMA 2 Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self):
        return LLaMA(
            src_vocab_size=381,
            emb_dim=16,
            norm_eps=1e-05,
            nheads=4,
            kvheads=2,
            nlayers=2,
            pad_id=0,
            hidden_grow_factor=2.6666666666666665,
            multiple_of=2,
            activation_fn="swish",
            p_dropout=0.0,
            max_expected_seq_len=4096,
            ntk_scaling=False,
        )


class TestLlama2GQA(ModelConsistencyTestSuite, LLaMA2GQAFixtures):
    """
    Test LLaMA2-GQA model consistency
    """

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]
