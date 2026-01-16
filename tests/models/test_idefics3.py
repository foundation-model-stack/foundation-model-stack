import pytest
import torch

from fms.models.idefics3 import Idefics3, Idefics3Config
from fms.models.llama import LLaMAConfig
from fms.models.siglip_vision import SiglipVisionConfig
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig


class Idefics3Fixtures(ConfigFixtureMixin, ModelFixtureMixin):
    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        # Keep this tiny so unit tests are fast and do not require external deps.
        vision_config = SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            nlayers=1,
            nheads=4,
            num_channels=3,
            image_size=32,
            patch_size=16,
            fused_weights=True,
        )

        text_config = LLaMAConfig(
            src_vocab_size=100,
            emb_dim=32,
            norm_eps=1e-5,
            nheads=4,
            kvheads=0,
            nlayers=1,
            pad_id=0,
            hidden_grow_factor=2.0,
            multiple_of=1,
            p_dropout=0.0,
            max_expected_seq_len=64,
            attn_bias=False,
            mlp_bias=False,
            tie_heads=False,
            fused_weights=True,
        )

        # With image_size=32 and patch_size=16 => 2x2 patches => 4 patch tokens.
        # connector_scale=1 keeps 4 image tokens so packing is easy to test.
        return Idefics3Config(
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=99,
            image_span_len=4,
            connector_scale=1,
            max_new_tokens=8,
        )

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: Idefics3Config):
        torch.manual_seed(123)
        model = Idefics3(config)
        model.reset_parameters()
        return model


class TestIdefics3(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    Idefics3Fixtures,
):
    @staticmethod
    def get_logits(f_out) -> torch.Tensor:
        return f_out["logits"]

    _get_signature_params = ["input_ids"]

    # Provide input_ids with exactly one contiguous image span of length image_span_len (4).
    _get_signature_input_ids = torch.tensor(
        [[1, 99, 99, 99, 99, 2, 3, 4]], dtype=torch.int64
    )
    _get_signature_optional_params = {
        "pixel_values": torch.zeros(1, 3, 32, 32),
        "attention_mask": torch.ones(1, 8, dtype=torch.int64),
    }
    _get_signature_logits_getter_fn = get_logits

    def test_config_passed_to_model_and_updated(self, model, config):
        model = type(model)(config=config, image_token_id=config.image_token_id + 1)
        assert model.get_config() is not config

        config.image_token_id = config.image_token_id + 1
        assert model.get_config().as_dict() == config.as_dict()

    def test_config_params_passed_as_kwargs_to_model(self, model, config):
        pytest.skip(
            "idefics3 uses nested configs for vision and text model, which get flattened with config.as_dict()"
        )

    def test_generate_right_padded_matches_unpadded(self, model, config):
        # Deterministic init for stable output.
        torch.manual_seed(123)

        # Single image span of 4 tokens (image_token_id=99).
        prompt = torch.tensor([[1, 99, 99, 99, 99, 2]], dtype=torch.int64)
        pixel_values = torch.zeros(1, 3, 32, 32)

        # Unpadded
        out_unpadded = model.generate(
            input_ids=prompt,
            pixel_values=pixel_values,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=3,
            eos_token_id=None,
        )

        # Right padded to a longer length
        pad_len = 4
        pad_id = config.text_config.pad_id
        padded = torch.cat(
            [prompt, torch.full((1, pad_len), pad_id, dtype=torch.int64)], dim=1
        )
        padded_mask = torch.cat(
            [torch.ones_like(prompt), torch.zeros((1, pad_len), dtype=torch.int64)],
            dim=1,
        )
        out_padded = model.generate(
            input_ids=padded,
            pixel_values=pixel_values,
            attention_mask=padded_mask,
            max_new_tokens=3,
            eos_token_id=None,
        )

        # `generate()` normalizes to left-padding internally, so the padded case will include
        # leading pad tokens. The suffix (prompt + new tokens) should match the unpadded output.
        assert torch.equal(out_padded[:, -out_unpadded.shape[1] :], out_unpadded)
