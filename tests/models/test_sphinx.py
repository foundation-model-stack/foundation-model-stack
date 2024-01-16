import pytest
from fms.models.sphinx import Sphinx, SphinxConfig
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)


class SphinxFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Sphinx Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: SphinxConfig):
        return Sphinx(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> SphinxConfig:
        return SphinxConfig(
            src_vocab_size=384,
            emb_dim=16,
            norm_eps=1e-5,
            nheads=8,
            nlayers=2,
            hidden_grow_factor=2.0,
            multiple_of=1,
            kvheads=4,
            activation_fn="swish",
            max_expected_seq_len=512,
            pad_id=0,
        )


class TestSphinx(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    SphinxFixtures,
):
    """
    Model Test Suite for Sphinx

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
