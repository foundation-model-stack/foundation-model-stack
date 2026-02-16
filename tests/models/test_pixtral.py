import pytest
import torch

from fms.models.pixtral_vision import PixtralVisionModel, PixtralVisionConfig

from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class PixtralFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Pixtral Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: PixtralVisionConfig):
        return PixtralVisionModel(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return PixtralVisionConfig(
            hidden_size=16,
            intermediate_size=64,
            nlayers=8,
            nheads=8,
            nchannels=3,
            image_size=280,
            patch_size=14,
            hidden_act="silu",
            layer_norm_eps=1e-5,
            rope_theta=10000.0,
            attention_dropout=0.0,
            fused_weights=True,
        )


class TestPixtral(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    PixtralFixtures,
):
    @staticmethod
    def get_last_hidden_state(f_out):
        return f_out[0]

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]
    pixel_values = [
        [[torch.arange(0, 1, 1 / 280).tolist() for _ in range(280)] for _ in range(3)]
    ]
    pixel_values = torch.tensor(pixel_values)  # [1, 3, 280, 280]
    # NOTE: image_sizes is actually required, but the current test utils
    # don't appear to support passing values required params, so we
    # pass it here.
    _get_signature_optional_params = {
        "pixel_values": pixel_values,
        "image_sizes": [(280, 280)],
    }
    _get_signature_logits_getter_fn = get_last_hidden_state

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(config=config, nlayers=config.nlayers + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify nlayers to the new value expected and check equivalence
        config.nlayers = config.nlayers + 1
        assert model.get_config().as_dict() == config.as_dict()
