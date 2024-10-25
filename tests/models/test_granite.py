import pytest
import torch

from fms.models.granite import Granite, GraniteConfig, GraniteHeadless
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class GraniteFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Granite Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: GraniteConfig):
        return Granite(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return GraniteConfig(
            src_vocab_size=384,
            emb_dim=16,
            norm_eps=1e-05,
            nheads=8,
            kvheads=8,
            nlayers=2,
            pad_id=0,
            hidden_grow_factor=3.125,
            multiple_of=2,
            activation_fn="swish",
            p_dropout=0.0,
            max_expected_seq_len=4096,
            ntk_scaling=False,
            linear_config={"linear_type": "torch_linear"},
            unfuse_strategy=None,
        )


class TestGranite(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    GraniteFixtures,
):
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

    @pytest.fixture
    def headless_model(self, model: Granite) -> GraniteHeadless:
        return model.base_model
