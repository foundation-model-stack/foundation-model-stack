import pytest

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
    def uninitialized_model(self, config: BambaConfig):
        return Bamba(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return BambaConfig(
            src_vocab_size=384,
            emb_dim=128,
            tie_heads=False,
            norm_eps=1e-05,
            kvheads=2,
            nlayers=2,
            nheads=8,
            use_bias=False,
            head_dim=64,
            n_groups=1,
            hidden_grow_factor=3.5,
            mamba_expand=2.0,
            state_size=128,
            conv_kernel=4,
            use_conv_bias=True,
            chunk_size=256,
            attn_layer_indices=[9, 18, 27],
            mamba_n_heads=4,
        )


class TestBamba(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    BambaFixtures,
):
    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(config=config, fused_weights=False)
        # check not same reference
        assert model.get_config() is not config

        # modify fused_weights to the new value expected and check equivalence
        config.fused_weights = False
        assert model.get_config().as_dict() == config.as_dict()

    @pytest.fixture
    def headless_model(self, model: Bamba) -> BambaHeadless:
        return model.base_model
