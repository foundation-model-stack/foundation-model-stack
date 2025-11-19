import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Unpack, Tuple

import torch
import torch.nn as nn

from fms import models
from fms.models.siglip_vision import SiglipVision, SiglipVisionConfig
from fms.models.granite import Granite, GraniteConfig
from fms.modules.linear import get_linear_type
from fms.modules.attention import AttentionKwargs
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig

# TODO: percolate ditributed_strategy, taking care of _no_split_modules from original transformers code

logger = logging.getLogger(__name__)

_granite_3_2_2b_text_config = GraniteConfig(
    src_vocab_size=49156,
    emb_dim=2048,
    norm_eps=1e-5,
    nheads=32,
    head_dim=64,
    kvheads=8,
    nlayers=40,
    hidden_grow_factor=8192 / 2048,
    max_expected_seq_len=131072,
    rope_theta=300000.0,
    pad_id=0,
    p_dropout=0.0,
    tie_heads=True,
    embedding_multiplier=12.0,
    logits_scaling=8.0,
    residual_multiplier=0.22,
    attention_multiplier=0.015625,
    fused_weights=True,
)

_granite_3_2_2b_vision_config = SiglipVisionConfig(
    hidden_size=1152,
    image_size=384,
    intermediate_size=4304,
    nheads=16,
    nlayers=27,
    patch_size=14,
    fused_weights=True,
)


_granite_3_2_2b_grid = [
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


@dataclass
class LlavaNextConfig(ModelConfig):
    # Defaults to Granite-vision-3.2-2b
    vision_config: SiglipVisionConfig = field(
        default_factory=lambda: _granite_3_2_2b_vision_config
    )
    text_config: GraniteConfig = field(
        default_factory=lambda: _granite_3_2_2b_text_config
    )
    image_token_index: int = 49155
    projector_hidden_act: str = "gelu"
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: list = field(default_factory=lambda: [-24, -20, -12, -1])
    image_grid_pinpoints: list = field(default_factory=lambda: _granite_3_2_2b_grid)
    multimodal_projector_bias: bool = True
    fused_weights: bool = True


class LlavaNextMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaNextConfig):
        super().__init__()
        num_feature_layers = (
            1
            if isinstance(config.vision_feature_layer, int)
            else len(config.vision_feature_layer)
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.emb_dim,
            bias=config.multimodal_projector_bias,
        )
        self.act = str_to_activation(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.emb_dim,
            config.text_config.emb_dim,
            bias=config.multimodal_projector_bias,
        )
        self.config = config

    # NOTE: HF doesn't do weight initialization for LlavaNextMultiModalProjector
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        if self.config.multimodal_projector_bias:
            nn.init.normal_(self.linear_1.bias, std=1e-6)
            nn.init.normal_(self.linear_2.bias, std=1e-6)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaNext(nn.Module):
    def __init__(
        self,
        config: Optional[LlavaNextConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(LlavaNext, self).__init__()

        if config is not None:
            self.config = config
        else:
            self.config = LlavaNextConfig()

        self.config = self.config.updated(**kwargs)

        if not self.config.fused_weights:
            self.config.text_config.fused_weights = False
            self.config.vision_config.fused_weights = False

        self.distributed_strategy = distributed_strategy

        if not isinstance(self.config.vision_config, SiglipVisionConfig):
            print(
                "FMS implementation of LlavaNext currently supports only Siglip vision model"
            )
        if not isinstance(self.config.text_config, GraniteConfig):
            print(
                "FMS implementation of LlavaNext currently supports only Granite language model"
            )

        # Only supporting granite text decoder encoder for now
        self.language_model = Granite(self.config.text_config)

        # Only supporting siglip vision encoder for now
        self.vision_tower = SiglipVision(self.config.vision_config)

        self.multi_modal_projector = LlavaNextMultiModalProjector(self.config)
        embed_std = 1 / math.sqrt(self.config.text_config.emb_dim)
        self.image_newline = nn.Parameter(
            torch.randn(self.config.text_config.emb_dim) * embed_std
        )
        self.vocab_size = self.config.text_config.src_vocab_size

    @classmethod
    def from_config(cls, config: LlavaNextConfig) -> "LlavaNext":
        return cls(config)

    def get_config(self) -> LlavaNextConfig:
        return self.config

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.image_newline.data)
        self.langauage_model.reset_parameters()
        self.vision_tower.reset_parameters()
        self.multi_modal_projector.reset_parameters()

    def post_init(self):
        self.language_model.post_init()
        self.vision_tower.post_init()

    def unpad_image(self, tensor: torch.Tensor, original_image_size: torch.Tensor):
        if not isinstance(original_image_size, (list, tuple)):
            original_size = original_image_size.tolist()
        original_height, original_width = original_size
        current_height, current_width = tensor.shape[1:]

        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = current_width / original_width
            new_height = int(round(original_height * scale_factor, 7))
            padding = (current_height - new_height) // 2
            unpadded_tensor = tensor[:, padding : current_height - padding, :]
        else:
            scale_factor = current_height / original_height
            new_width = int(round(original_width * scale_factor, 7))
            padding = (current_width - new_width) // 2
            unpadded_tensor = tensor[:, :, padding : current_width - padding]

        return unpadded_tensor

    # TODO: fix graph break in the HF impl here
    def select_best_resolution(
        self,
        original_image_size: torch.Tensor,
        possible_resolutions: list[Tuple[int, int]],
    ):
        if not isinstance(original_image_size, (list, tuple)):
            original_size = original_image_size.tolist()

        original_height, original_width = original_size
        best_fit = None
        max_effective_resolution = 0
        min_wasted_resolution = float("inf")

        for height, width in possible_resolutions:
            scale = min(width / original_width, height / original_height)
            downscaled_width, downscaled_height = (
                int(original_width * scale),
                int(original_height * scale),
            )
            effective_resolution = min(
                downscaled_width * downscaled_height, original_width * original_height
            )
            wasted_resolution = (width * height) - effective_resolution

            if effective_resolution > max_effective_resolution or (
                effective_resolution == max_effective_resolution
                and wasted_resolution < min_wasted_resolution
            ):
                max_effective_resolution = effective_resolution
                min_wasted_resolution = wasted_resolution
                best_fit = (height, width)

        return best_fit

    def image_size_to_num_patches(
        self,
        image_size: torch.Tensor,
        grid_pinpoints: list[Tuple[int, int]],
        patch_size: int,
    ):
        height, width = self.select_best_resolution(image_size, grid_pinpoints)
        num_patches = 1 + math.ceil(height / patch_size) * math.ceil(width / patch_size)
        return num_patches

    # HF impl
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ):
        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            self.image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [
                pix_val[:num_patch]
                for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(
                f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions"
            )

        _, _, image_features = self.vision_tower(
            pixel_values, output_hidden_states=True
        )
        if isinstance(self.config.vision_feature_layer, int):
            selected_image_feature = image_features[self.config.vision_feature_layer]
        else:
            hs_pool = [
                image_features[layer_idx]
                for layer_idx in self.config.vision_feature_layer
            ]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        if self.config.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]

        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        return image_features

    def pack_image_features(
        self,
        image_features: list[torch.Tensor],
        image_sizes: torch.Tensor,
        image_newline: Optional[torch.Tensor] = None,
    ):
        new_image_features = []

        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = (
                    self.config.vision_config.image_size
                    // self.config.vision_config.patch_size
                )

                patch_height, patch_width = self.select_best_resolution(
                    image_sizes[image_idx], self.config.image_grid_pinpoints
                )
                num_patch_height = patch_height // self.config.vision_config.image_size
                num_patch_width = patch_width // self.config.vision_config.image_size

                image_feature = image_feature.view(
                    num_patch_height, num_patch_width, height, width, -1
                )
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = self.unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat(
                        (image_feature, image_newline[None].to(image_feature)), dim=0
                    )
            new_image_features.append(image_feature)
        return torch.cat(new_image_features, dim=0)

    def forward(
        self,
        input_ids_or_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: Optional[bool] = False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        outputs = self.language_model(
            input_ids_or_embeds,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            use_cache=use_cache,
            **attn_kwargs,
        )
        return outputs

    def prepare_inputs_for_generation(
        self,
        iteration,
        input_ids,
        kwargs,
    ):
        # Use with arg `prepare_model_inputs_hook=model.prepare_inputs_for_generation` when calling generate()

        if kwargs["use_cache"] and iteration > 0:
            # No need to process image data again in cached decoding stage.
            input_ids = self.language_model.base_model.embedding(input_ids)
            return input_ids, kwargs

        pixel_values = kwargs.get("pixel_values")
        image_sizes = kwargs.get("image_sizes")

        # No image data to pre-process
        if pixel_values is None or pixel_values.size(0) == 0:
            return input_ids, kwargs

        inputs = kwargs.get("inputs")
        if input_ids is None and inputs is None:
            raise ValueError("input_ids and inputs can't both be None")

        # embedded inputs supersede input_ids
        if inputs is None:
            inputs = self.language_model.base_model.embedding(input_ids)

        image_features = self.get_image_features(
            pixel_values,
            image_sizes,
        )

        image_features = self.pack_image_features(
            image_features,
            image_sizes,
            image_newline=self.image_newline,
        )

        special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs).to(inputs.device)
        image_features = image_features.to(inputs.device, inputs.dtype)
        inputs = inputs.masked_scatter(special_image_mask, image_features)
        return inputs, kwargs


_granite_vision_3_2_2b_config = LlavaNextConfig()

_architecture_name = "llava_next"


def _llava_next_factory_factory(config):
    def factory(**kwargs):
        return LlavaNext(config, **kwargs)

    return factory


models.register_model(
    _architecture_name,
    "granite_vision_3_2_2b",
    _llava_next_factory_factory(_granite_vision_3_2_2b_config),
)


def _weight_fusion(
    input_sd: Mapping, model_config: Optional[LlavaNextConfig] = None, **kwargs
):
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        new_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
            serialization._attn_unfused_to_fused_step(new_sd)
        )
    return new_sd


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        # vision
        (r"vision_tower\.vision_model\.head", "vision_tower.head"),
        (r"vision_tower\.vision_model\.encoder", "vision_tower.base_model.encoder"),
        (
            r"vision_tower\.vision_model\.embeddings",
            "vision_tower.base_model.embeddings",
        ),
        (
            r"vision_tower\.vision_model\.post_layernorm",
            "vision_tower.base_model.post_layernorm",
        ),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.out_proj", "attn.dense"),
        (r"mlp\.fc1", "mlp.w1"),
        (r"mlp\.fc2", "mlp.w2"),
        # language
        (r"language_model\.lm_head\.weight", "language_model.head.weight"),
        (
            r"language_model.model.embed_tokens.weight",
            "language_model.base_model.embedding.weight",
        ),
        (r"language_model.model.norm", "language_model.base_model.dec_norm"),
        (r"language_model.model.layers", "language_model.base_model.layers"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


# From Granite model
def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    if "int8" in linear_type:
        # quantize_weight is fms-model-optimizer identifier of weight clip values
        return ["weight", "bias", "quantize_weight"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


# From Granite model
def _hf_to_fms_rope(
    input_sd: Mapping[str, Any],
    model_config=None,
    **kwargs,
) -> Mapping[str, Any]:
    new_sd = {}
    if model_config:
        model_config = model_config.text_config

    if model_config:
        head_size = model_config.emb_dim // model_config.nheads
        linear_type_str = "torch_linear"
        if model_config.linear_config:
            linear_type_str = get_linear_type(
                model_config.linear_config,
                module_name=None,  # if callable, linear_type should return default str
            )
    else:
        logger.warning("Missing model_config, assuming defaults for head_size")
        head_size = 128  # Good default for most models
        linear_type_str = "torch_linear"

    rope_params = _get_rope_params(linear_type_str)
    trans_required_pattern = re.compile(
        f"language_model.base_model.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})"
    )
    for name, param in input_sd.items():
        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters for Llama models
        # This transpose is due to the different implementation of RoPE in
        # HF and FMS. While FMS follows the original RoPE paper
        # (https://arxiv.org/abs/2104.09864), HF has its own implementation
        # that doesn't respect the order of outputs. This is OK as long as you
        # rearrange the weights of the query and key projections, as the
        # combination projection + RoPE ends up producing the same outputs.
        # Therefore, to make FMS produce the correct order of outputs when
        # loading from an HF checkpoint, we need to undo the transformation
        # that HF does from the original Meta weights
        is_gptq_2d_qparam = "gptq" in linear_type_str and param.dim() == 2
        if bool(trans_required_pattern.match(name)) and param.numel() > 1:
            temp = param
            if is_gptq_2d_qparam:
                # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
                # and are fully transposed before & after process.
                # GPTQ scales and qzeros are also transposed accordingly
                temp = temp.transpose(0, 1)
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # 1-dim parameters
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            if is_gptq_2d_qparam:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)


serialization.register_adapter_step(
    _architecture_name,
    "weight_expansion_for_mismatched_head_dim",
    serialization._weight_expansion_for_mismatched_head_dim,  # type: ignore[arg-type]
)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "weight_fusion"],
)
