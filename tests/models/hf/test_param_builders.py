from __future__ import annotations

from fms.models.hf.config_utils.param_builders import (
    build_siglip_idefics3_vision_params,
    build_siglip_vision_params,
)


class _DummySiglipVisionConfig:
    def __init__(
        self,
        *,
        image_size: int = 512,
        patch_size: int = 16,
        model_type: str | None = None,
        max_image_size: int | None = None,
    ) -> None:
        # Keep these minimal but consistent with what the param builders expect.
        self.hidden_size = 128
        self.intermediate_size = 256
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.num_channels = 3
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = "gelu"
        self.layer_norm_eps = 1e-5
        self.attention_dropout = 0.0

        # Optional attributes present on some HF configs.
        self.model_type = model_type
        if max_image_size is not None:
            self.max_image_size = max_image_size


class _DummyConfig:
    def __init__(
        self, *, vision_config: _DummySiglipVisionConfig, model_type: str | None
    ):
        self.vision_config = vision_config
        self.model_type = model_type


def test_build_siglip_vision_params_defaults_navit_false_when_attr_missing():
    cfg = _DummyConfig(vision_config=_DummySiglipVisionConfig(), model_type="siglip")

    params = build_siglip_vision_params(cfg)

    assert params["use_navit_position_buckets"] is False


def test_build_siglip_idefics3_vision_params_enables_navit_for_smolvlm_max_image_size():
    cfg = _DummyConfig(
        vision_config=_DummySiglipVisionConfig(model_type=None, max_image_size=1024),
        model_type="smolvlm",
    )

    params = build_siglip_idefics3_vision_params(cfg)

    assert params["use_navit_position_buckets"] is True


def test_build_siglip_idefics3_vision_params_does_not_enable_navit_for_non_idefics3_parent():
    cfg = _DummyConfig(
        vision_config=_DummySiglipVisionConfig(model_type=None, max_image_size=1024),
        model_type="siglip",
    )

    params = build_siglip_idefics3_vision_params(cfg)

    assert params["use_navit_position_buckets"] is False
