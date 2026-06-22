import pytest
import torch

from fms.models.siglip_vision import SiglipVisionConfig
from fms.models.granite import GraniteConfig
from fms.models.llava_next import LlavaNext, LlavaNextConfig
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class LlavaNextFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base LlavaNext Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: LlavaNextConfig):
        return LlavaNext(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        _text_config = GraniteConfig(
            src_vocab_size=384,
            emb_dim=16,
            norm_eps=1e-5,
            nheads=8,
            head_dim=2,
            kvheads=8,
            nlayers=2,
            hidden_grow_factor=4,
            max_expected_seq_len=4096,
            pad_id=0,
            p_dropout=0.0,
            tie_heads=True,
            embedding_multiplier=12.0,
            logits_scaling=8.0,
            residual_multiplier=0.22,
            attention_multiplier=0.015625,
            fused_weights=True,
        )

        _vision_config = SiglipVisionConfig(
            hidden_size=16,
            image_size=384,
            intermediate_size=64,
            nheads=8,
            nlayers=8,
            patch_size=14,
            fused_weights=True,
        )

        _image_grid = [
            [384, 384],
            [384, 768],
            [384, 1152],
            [384, 1536],
            [384, 1920],
            [384, 2304],
            [384, 2688],
            [384, 3072],
            [384, 3456],
            [384, 3840],
            [768, 384],
            [768, 768],
            [768, 1152],
            [768, 1536],
            [768, 1920],
            [1152, 384],
            [1152, 768],
            [1152, 1152],
            [1536, 384],
            [1536, 768],
            [1920, 384],
            [1920, 768],
            [2304, 384],
            [2688, 384],
            [3072, 384],
            [3456, 384],
            [3840, 384],
        ]

        return LlavaNextConfig(
            vision_config=_vision_config,
            text_config=_text_config,
            image_token_index=383,
            projector_hidden_act="gelu",
            vision_feature_select_strategy="full",
            vision_feature_layer=[-4, -2],
            image_grid_pinpoints=_image_grid,
            multimodal_projector_bias=True,
            fused_weights=True,
        )


class TestLlavaNext(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    LlavaNextFixtures,
):
    @staticmethod
    def get_logits(f_out):
        return f_out[0]

    # sample image size, pixel values and input_ids for testing
    image_sizes = torch.tensor([[686, 960]])
    pixel_values = [
        [
            [
                [torch.arange(0, 1, 1 / 384).tolist() for _ in range(384)]
                for _ in range(3)
            ]
            for _ in range(7)
        ]
    ]
    pixel_values = torch.tensor(pixel_values)  # shape [1, 7, 3, 384, 384]
    input_ids = torch.arange(380).unsqueeze(0)

    _get_signature_params = ["input_ids_or_embeds"]
    _get_signature_input_ids = input_ids
    _get_signature_optional_params = {
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
        "last_n_tokens": 1,
    }
    _get_signature_logits_getter_fn = get_logits

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(
            config=config, vision_feature_layer=config.vision_feature_layer[1:]
        )
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.vision_feature_layer = config.vision_feature_layer[1:]
        assert model.get_config().as_dict() == config.as_dict()

    def test_config_params_passed_as_kwargs_to_model(self, model, config):
        pytest.skip(
            "llava_next uses nested configs for vision and text model, which get flattened with config.as_dict()"
        )


class TestLlavaNextVisionOnly:
    """Tests for LlavaNext vision_only=True mode."""

    @pytest.fixture
    def config(self):
        _text_config = GraniteConfig(
            src_vocab_size=384,
            emb_dim=16,
            norm_eps=1e-5,
            nheads=8,
            head_dim=2,
            kvheads=8,
            nlayers=2,
            hidden_grow_factor=4,
            max_expected_seq_len=4096,
            pad_id=0,
            p_dropout=0.0,
            tie_heads=True,
            embedding_multiplier=12.0,
            logits_scaling=8.0,
            residual_multiplier=0.22,
            attention_multiplier=0.015625,
            fused_weights=True,
        )
        _vision_config = SiglipVisionConfig(
            hidden_size=16,
            image_size=384,
            intermediate_size=64,
            nheads=8,
            nlayers=8,
            patch_size=14,
            fused_weights=True,
        )
        return LlavaNextConfig(
            vision_config=_vision_config,
            text_config=_text_config,
            image_token_index=383,
            projector_hidden_act="gelu",
            vision_feature_select_strategy="full",
            vision_feature_layer=[-4, -2],
            multimodal_projector_bias=True,
            fused_weights=True,
        )

    def test_vision_only_skips_language_model(self, config):
        model = LlavaNext(config, vision_only=True)
        assert model._vision_only is True
        assert not hasattr(model, "language_model")
        assert hasattr(model, "text_embedding")
        assert hasattr(model, "vision_tower")
        assert hasattr(model, "multi_modal_projector")

    def test_vision_only_false_has_language_model(self, config):
        model = LlavaNext(config, vision_only=False)
        assert model._vision_only is False
        assert hasattr(model, "language_model")
        assert not hasattr(model, "text_embedding")

    def test_vision_only_text_embedding_shape(self, config):
        model = LlavaNext(config, vision_only=True)
        assert model.text_embedding.num_embeddings == config.text_config.src_vocab_size
        assert model.text_embedding.embedding_dim == config.text_config.emb_dim

    def test_vision_only_get_text_embeddings(self, config):
        model = LlavaNext(config, vision_only=True)
        input_ids = torch.arange(10).unsqueeze(0)
        embeds = model._get_text_embeddings(input_ids)
        assert embeds.shape == (1, 10, config.text_config.emb_dim)

    def test_vision_only_prepare_inputs_no_image(self, config):
        """With no pixel_values, prepare_inputs returns text embeddings only."""
        model = LlavaNext(config, vision_only=True)
        model.eval()
        input_ids = torch.arange(10).unsqueeze(0)
        kwargs = {"use_cache": False, "attn_name": "sdpa_causal"}
        embeds, _ = model.prepare_inputs_for_generation(0, input_ids, kwargs)
        assert embeds.shape == (1, 10, config.text_config.emb_dim)

    def test_vision_only_prepare_inputs_missing_use_cache(self, config):
        """use_cache absent from kwargs must not raise a KeyError."""
        model = LlavaNext(config, vision_only=True)
        model.eval()
        input_ids = torch.arange(10).unsqueeze(0)
        kwargs = {"attn_name": "sdpa_causal"}
        embeds, _ = model.prepare_inputs_for_generation(0, input_ids, kwargs)
        assert embeds.shape == (1, 10, config.text_config.emb_dim)

    def test_vision_only_state_dict_has_no_language_model(self, config):
        model = LlavaNext(config, vision_only=True)
        keys = list(model.state_dict().keys())
        assert not any(k.startswith("language_model.") for k in keys)
        assert any(k.startswith("vision_tower.") for k in keys)
        assert any(k.startswith("multi_modal_projector.") for k in keys)
        assert any(k.startswith("text_embedding.") for k in keys)

    def test_vision_only_reset_parameters_does_not_access_language_model(self, config):
        """reset_parameters must not raise AttributeError about language_model being absent."""
        model = LlavaNext(config, vision_only=True)
        # Patch vision_tower.reset_parameters to avoid unrelated recursion in siglip
        import unittest.mock as mock

        with mock.patch.object(model.vision_tower, "reset_parameters"):
            model.reset_parameters()  # should not raise AttributeError

    def test_vision_only_hf_prefixes_class_attribute(self):
        assert hasattr(LlavaNext, "VISION_ONLY_HF_PREFIXES")
        prefixes = LlavaNext.VISION_ONLY_HF_PREFIXES
        assert any("vision_tower" in p for p in prefixes)
        assert any("multi_modal_projector" in p for p in prefixes)
        assert any("embed_tokens" in p for p in prefixes)
