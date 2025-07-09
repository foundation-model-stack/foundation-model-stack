import pytest
import torch

from fms.models.pixtral import PixtralVisionConfig
from fms.models.mistral import MistralConfig
from fms.models.llava import Llava, LlavaConfig
from fms.testing._internal.model_test_suite import (
    ConfigFixtureMixin,
    ModelCompileTestSuite,
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelFixtureMixin,
)
from fms.utils.config import ModelConfig

# TODO: reduce dims
_text_config = MistralConfig(
    src_vocab_size=384,
    emb_dim=16,
    norm_eps=1e-5,
    nheads=8,
    kvheads=8,
    nlayers=2,
    hidden_grow_factor=4,
    max_expected_seq_len=4096,
    rope_base=1000000000.0,
    sliding_window=None,
    head_dim=2,
)

_vision_config = PixtralVisionConfig(
    hidden_size=16,
    image_size=384,
    intermediate_size=64,
    nheads=8,
    nlayers=8,
    nchannels=3,
    patch_size=16,
    rope_theta=10000.0,
    hidden_act="silu",
)


class LlavaFixtures(ConfigFixtureMixin, ModelFixtureMixin):
    """
    Base Llava Fixtures that can be re-used for other purposes

    This will include the config and model signatures
    """

    @pytest.fixture(scope="class", autouse=True)
    def uninitialized_model(self, config: LlavaConfig):
        return Llava(config)

    @pytest.fixture(scope="class", autouse=True)
    def config(self) -> ModelConfig:
        return LlavaConfig(
            vision_config=_vision_config,
            text_config=_text_config,
            image_token_index=10,
            projector_hidden_act="gelu",
            vision_feature_select_strategy="full",
            vision_feature_layer=-1,
            multimodal_projector_bias=True,
            fused_weights=True,
        )


class TestLlava(
    ModelConfigTestSuite,
    ModelConsistencyTestSuite,
    ModelCompileTestSuite,
    LlavaFixtures,
):
    @staticmethod
    def get_logits(f_out):
        return f_out[0]

    # sample image size, pixel values and input_ids for testing
    image_sizes = torch.tensor([[384, 384]])
    pixel_values = [
        [[torch.arange(0, 1, 1 / 384).tolist() for _ in range(384)] for _ in range(3)]
    ]
    pixel_values = torch.tensor(pixel_values)  # shape [1, 3, 384, 384]
    input_ids = torch.arange(380).unsqueeze(0)

    _get_signature_params = ["input_ids"]
    _get_signature_input_ids = input_ids
    _get_signature_optional_params = {
        "pixel_values": pixel_values,
        "image_sizes": image_sizes,
        "only_last_token": True,
    }
    _get_signature_logits_getter_fn = get_logits

    def test_config_passed_to_model_and_updated(self, model, config):
        """test model constructor appropriately merges any passed kwargs into the config without mutating the original config"""
        model = type(model)(
            config=config, image_token_index=config.image_token_index + 1
        )
        # check not same reference
        assert model.get_config() is not config

        # modify pad_id to the new value expected and check equivalence
        config.image_token_index = config.image_token_index + 1
        assert model.get_config().as_dict() == config.as_dict()

    def test_config_params_passed_as_kwargs_to_model(self, model, config):
        pytest.skip(
            "llava uses nested configs for vision and text model, which get flattened with config.as_dict()"
        )

    def test_model_compile_no_graph_breaks(self, model):
        pytest.skip(
            "data dependent graph break exists in original implementation of pixtral; needs to be fixed"
        )
