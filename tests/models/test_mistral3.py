import pytest
import torch

from fms.models.pixtral_vision import PixtralVisionConfig
from fms.models.mistral import MistralConfig
from fms.models.mistral3 import Mistral3, Mistral3Config
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class Mistral3Fixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Mistral3 Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: Mistral3Config):
        return Mistral3(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        # Text / vision configs are essentially those used for the
        # mistral llm and pixtral encoder tests
        _text_config = MistralConfig(
            src_vocab_size=384,
            nheads=8,
            nlayers=2,
            hidden_grow_factor=3.5,
            multiple_of=2,
            tie_heads=False,
            p_dropout=0.0,
            activation_fn="swish",
            emb_dim=16,
            head_dim=4096 // 32,
            max_expected_seq_len=4096,
            kvheads=2,
            norm_eps=1e-05,
            sliding_window=4000,
            rope_base=100_0000.0,
            fused_weights=True,
            pad_id=0,
        )

        _vision_config = PixtralVisionConfig(
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

        return Mistral3Config(
            vision_config=_vision_config,
            text_config=_text_config,
            spatial_merge_size=2,
            image_token_index=10,
            vision_feature_layer=-1,
        )


class TestMistral3(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    Mistral3Fixtures,
):
    @staticmethod
    def get_logits(f_out):
        return f_out[0]

    pixel_values = [
        [[torch.arange(0, 1, 1 / 280).tolist() for _ in range(280)] for _ in range(3)]
    ]
    input_ids = torch.arange(380).unsqueeze(0)
    pixel_values = torch.tensor(pixel_values)  # [1, 3, 280, 280]

    _get_signature_params = ["input_ids_or_embeds"]
    _get_signature_input_ids = input_ids
    _get_signature_optional_params = {
        "pixel_values": pixel_values,
        "image_sizes": [(280, 280)],
        "last_n_tokens": 1,
    }

    _get_signature_logits_getter_fn = get_logits

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(
            config=config,
            vision_feature_layer=config.vision_feature_layer - 1,
        )
        # check not same reference
        assert model.get_config() is not config

        # modify feature layer to the new value expected and check equivalence
        config.vision_feature_layer = config.vision_feature_layer - 1
        assert model.get_config().as_dict() == config.as_dict()
