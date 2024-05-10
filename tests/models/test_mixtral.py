import pytest

from fms.models.mixtral import Mixtral, MixtralConfig
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class MixtralFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Mixtral Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: MixtralConfig):
        model = Mixtral(config)
        model.reset_parameters()
        return model

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return MixtralConfig(
            src_vocab_size=384,
            dim=16,
            norm_eps=1e-05,
            nheads=4,
            kvheads=1,
            nlayers=2,
            hidden_dim=56,
            num_experts=8,
            top_k_experts=2,
        )


class TestMixtral(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    MixtralFixtures,
):
    """
    Model Test Suite for Mixtral

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
