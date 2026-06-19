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
            head_dim=64,
            max_expected_seq_len=1024,
            kvheads=2,
            norm_eps=1e-05,
            sliding_window=None,
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": 1000.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 1.0,
                "original_max_position_embeddings": 512,
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
            nlayers=2,
            nheads=8,
            nchannels=3,
            image_size=280,
            patch_size=14,
            hidden_act="silu",
            layer_norm_eps=1e-5,
            rope_theta=1000.0,
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


class TestMinistral3VisionOnly:
    """Tests for Ministral3 vision_only=True mode."""

    @pytest.fixture
    def config(self):
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
            head_dim=64,
            max_expected_seq_len=1024,
            kvheads=2,
            norm_eps=1e-05,
            sliding_window=None,
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": 1000.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 1.0,
                "original_max_position_embeddings": 512,
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
            nlayers=2,
            nheads=8,
            nchannels=3,
            image_size=280,
            patch_size=14,
            hidden_act="silu",
            layer_norm_eps=1e-5,
            rope_theta=1000.0,
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

    def test_vision_only_skips_language_model(self, config):
        model = Ministral3(config, vision_only=True)
        assert model._vision_only is True
        assert not hasattr(model, "language_model")
        assert hasattr(model, "text_embedding")
        assert hasattr(model, "vision_tower")
        assert hasattr(model, "multi_modal_projector")

    def test_vision_only_false_has_language_model(self, config):
        model = Ministral3(config, vision_only=False)
        assert model._vision_only is False
        assert hasattr(model, "language_model")
        assert not hasattr(model, "text_embedding")

    def test_vision_only_text_embedding_shape(self, config):
        model = Ministral3(config, vision_only=True)
        assert model.text_embedding.num_embeddings == config.text_config.src_vocab_size
        assert model.text_embedding.embedding_dim == config.text_config.emb_dim

    def test_vision_only_get_text_embeddings(self, config):
        model = Ministral3(config, vision_only=True)
        input_ids = torch.arange(10).unsqueeze(0)
        embeds = model._get_text_embeddings(input_ids, None)
        assert embeds.shape == (1, 10, config.text_config.emb_dim)

    def test_vision_only_prepare_inputs_for_generation_no_image(self, config):
        """With no pixel_values, prepare_inputs returns text embeddings only."""
        model = Ministral3(config, vision_only=True)
        model.eval()
        input_ids = torch.arange(10).unsqueeze(0)
        kwargs = {"attn_name": "sdpa_causal"}
        embeds, _ = model.prepare_inputs_for_generation(0, input_ids, kwargs)
        assert embeds.shape == (1, 10, config.text_config.emb_dim)

    def test_vision_only_state_dict_has_no_language_model(self, config):
        model = Ministral3(config, vision_only=True)
        keys = list(model.state_dict().keys())
        assert not any(k.startswith("language_model.") for k in keys)
        assert any(k.startswith("vision_tower.") for k in keys)
        assert any(k.startswith("multi_modal_projector.") for k in keys)
        assert any(k.startswith("text_embedding.") for k in keys)

    def test_vision_only_reset_parameters_does_not_access_language_model(self, config):
        """reset_parameters must not raise AttributeError about language_model being absent."""
        model = Ministral3(config, vision_only=True)
        # Patch vision_tower.reset_parameters to avoid unrelated recursion in pixtral
        import unittest.mock as mock
        with mock.patch.object(model.vision_tower, "reset_parameters"):
            model.reset_parameters()  # should not raise AttributeError

    def test_vision_only_inherits_get_text_embeddings_from_mistral3(self, config):
        """Ministral3 uses Mistral3._get_text_embeddings — verify the MRO dispatches correctly."""
        from fms.models.mistral3 import Mistral3
        model = Ministral3(config, vision_only=True)
        assert type(model)._get_text_embeddings is Mistral3._get_text_embeddings
