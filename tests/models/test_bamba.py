import pytest
import torch

from fms.models.bamba import Bamba, BambaConfig, BambaHeadless
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig

class BambaFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Granite Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: GraniteConfig):
        return Bamba(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return BambaConfig(
            src_vocab_size=384,
            emb_dim=16,
            norm_eps=1e-05,
            nheads=8,
            kvheads=8,
            nlayers=4,
            hidden_grow_factor=2.0,
            multiple_of=2,
            activation_fn="swish",
            p_dropout=0.0,
            max_expected_seq_len=4096,
            ntk_scaling=False,
            linear_config={"linear_type": "torch_linear"},
            fused_weights=True,
            attn_layer_indices={1}
        )

class TestBamba(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    BambaFixtures,
):
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(config=config, fused_weights=False)
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.fused_weights = True
        assert model.get_config().as_dict() == config.as_dict()

    @pytest.fixture
    def headless_model(self, model: Bamba) -> BambaHeadless:
        return model.base_model