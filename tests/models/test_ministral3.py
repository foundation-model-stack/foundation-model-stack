import pytest
import torch

from fms.models.pixtral_vision import PixtralVisionConfig
from fms.models.ministral3 import (
    Ministral3,
    Ministral3Config,
    Ministral3TextConfig,
)
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class Ministral3Fixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Ministral3 Fixtures that can be re-used for other purposes

    This will include the config and model signatures for the multimodal variant
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: Ministral3Config):
        return Ministral3(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        # Text / vision configs are essentially those used for the
        # ministral3 llm and pixtral encoder tests
        _text_config = Ministral3TextConfig(
            src_vocab_size=384,
            nheads=8,
            nlayers=2,
            hidden_grow_factor=3.5,
            multiple_of=2,
            tie_heads=False,
            p_dropout=0.0,
            activation_fn="silu",
            emb_dim=16,
            head_dim=128,
            max_expected_seq_len=4096,
            kvheads=2,
            norm_eps=1e-05,
            sliding_window=None,
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": 100_0000.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 1.0,
                "original_max_position_embeddings": 4096,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "llama_4_scaling_beta": 0.1,
            },
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

        return Ministral3Config(
            vision_config=_vision_config,
            text_config=_text_config,
            spatial_merge_size=2,
            image_token_index=10,
            vision_feature_layer=-1,
        )


class TestMinistral3(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    Ministral3Fixtures,
):
    """Test suite for Ministral3 (multimodal model with vision)"""

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


class TestMinistral3Vision(Ministral3Fixtures):
    """Test suite specifically for Ministral3 vision capabilities (Pixtral integration)"""

    def test_vision_tower_exists(self, model):
        """Test that the vision tower is properly initialized"""
        assert hasattr(model, "vision_tower")
        assert model.vision_tower is not None

    def test_multimodal_projector_exists(self, model):
        """Test that the multimodal projector is properly initialized"""
        assert hasattr(model, "multi_modal_projector")
        assert model.multi_modal_projector is not None

    def test_vision_config_propagation(self, model, config):
        """Test that vision config is properly propagated to the vision tower"""
        assert model.vision_tower.config.hidden_size == config.vision_config.hidden_size
        assert model.vision_tower.config.nlayers == config.vision_config.nlayers
        assert model.vision_tower.config.nheads == config.vision_config.nheads

    def test_text_config_propagation(self, model, config):
        """Test that text config is properly propagated to the language model"""
        assert model.language_model.config.emb_dim == config.text_config.emb_dim
        assert model.language_model.config.nlayers == config.text_config.nlayers
        assert model.language_model.config.nheads == config.text_config.nheads

    def test_fused_weights_propagation(self, config):
        """Test that fused_weights setting propagates correctly"""
        # Test with fused_weights=False
        config_unfused = Ministral3Config(
            vision_config=config.vision_config,
            text_config=config.text_config,
            fused_weights=False,
        )
        model_unfused = Ministral3(config_unfused)
        assert not model_unfused.config.text_config.fused_weights
        assert not model_unfused.config.vision_config.fused_weights

        # Test with fused_weights=True (default)
        config_fused = Ministral3Config(
            vision_config=config.vision_config,
            text_config=config.text_config,
            fused_weights=True,
        )
        model_fused = Ministral3(config_fused)
        assert model_fused.config.text_config.fused_weights
        assert model_fused.config.vision_config.fused_weights


# Made with Bob
