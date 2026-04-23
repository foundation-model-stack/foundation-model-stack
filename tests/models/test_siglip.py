import pytest
import torch

from fms.models.siglip_vision import SiglipVision, SiglipVisionConfig
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class SiglipFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Siglip Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: SiglipVisionConfig):
        return SiglipVision(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=64,
            nlayers=8,
            nheads=8,
            num_channels=3,
            image_size=224,
            patch_size=16,
            hidden_act="gelu-tanh",
            layer_norm_eps=1e-6,
            attention_dropout=0.0,
            fused_weights=True,
        )


class TestSiglip(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    SiglipFixtures,
):
    @staticmethod
    def get_last_hidden_state(f_out):
        return f_out[0]

    # x is the main parameter for this model which is the input tensor
    _get_signature_params = ["x"]
    pixel_values = [
        [[torch.arange(0, 1, 1 / 224).tolist() for _ in range(224)] for _ in range(3)]
    ]
    pixel_values = torch.tensor(pixel_values)  # shape [1, 3, 224, 224]
    _get_signature_optional_params = {"pixel_values": pixel_values}
    _get_signature_logits_getter_fn = get_last_hidden_state

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(config=config, nlayers=config.nlayers + 1)
        # check not same reference
        assert model.get_config() is not config

        # modify nlayers to the new value expected and check equivalence
        config.nlayers = config.nlayers + 1
        assert model.get_config().as_dict() == config.as_dict()


def test_siglip_embeddings_ignores_patch_attention_mask_when_navit_disabled():
    """
    Regression test: if a caller passes a patch_attention_mask while NaViT bucketing is disabled, we should
    still use the standard positional embedding path (patch_attention_mask must not silently enable NaViT).
    """
    from fms.models.siglip_vision import SiglipVisionEmbeddings

    config = SiglipVisionConfig(
        hidden_size=8,
        image_size=32,
        patch_size=16,
        use_navit_position_buckets=False,
    )
    embeddings = SiglipVisionEmbeddings(config)

    torch.manual_seed(0)
    pixel_values = torch.randn(1, 3, 32, 32)
    patch_attention_mask = torch.tensor([[[True, True], [False, False]]])

    out_no_mask = embeddings(pixel_values, patch_attention_mask=None)
    out_with_mask = embeddings(pixel_values, patch_attention_mask=patch_attention_mask)

    torch.testing.assert_close(out_no_mask, out_with_mask)
