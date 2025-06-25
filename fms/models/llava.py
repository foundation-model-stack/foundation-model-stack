import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Unpack, Tuple

import pdb
import torch
import torch.nn as nn

from fms import models
from fms.models.pixtral import PixtralVision, PixtralVisionConfig
from fms.models.mistral import Mistral, MistralConfig
from fms.modules.linear import get_linear_type
from fms.modules.attention import AttentionKwargs
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig

# TODO: percolate ditributed_strategy, taking care of _no_split_modules from original transformers code

logger = logging.getLogger(__name__)

_text_config = MistralConfig(
    src_vocab_size=131072,
    emb_dim=5120,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=40,
    hidden_grow_factor=14336 / 5120,
    max_expected_seq_len=1024000,   #TODO: might need to change
    rope_base=1000000000.0,
    sliding_window=None,
    fused_weights=False, #TODO: revert
    head_dim=128, #TODO: might need to fix emb_kq and emb_v to head_dim in Mistral
)

_vision_config = PixtralVisionConfig(
    hidden_size=1024,
    image_size=1024,
    intermediate_size=4096,
    nheads=16,
    nlayers=24,
    nchannels=3,
    patch_size=16,
    rope_theta=10000.0,
    hidden_act="silu",
)


@dataclass
class LlavaConfig(ModelConfig):
    vision_config: PixtralVisionConfig = field(
        default_factory=lambda: _vision_config
    )
    text_config: MistralConfig = field(
        default_factory=lambda: _text_config
    )
    image_token_index: int = 10
    projector_hidden_act: str = "gelu"
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: int = -1
    multimodal_projector_bias: bool = True
    fused_weights: bool = True


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
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


class Llava(nn.Module):
    def __init__(
        self,
        config: Optional[LlavaConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(Llava, self).__init__()

        if config is not None:
            self.config = config
        else:
            self.config = LlavaConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        if not isinstance(self.config.vision_config, PixtralVisionConfig):
            print(
                "FMS implementation of Llava currently supports only Pixtral vision model"
            )
        if not isinstance(self.config.text_config, MistralConfig) and not isinstance(self.config.text_config, GraniteConfig):
            print(
                "FMS implementation of Llava currently supports only Mistral and Granite language models"
            )

        self.language_model = Mistral(self.config.text_config)
        self.vision_tower = PixtralVision(self.config.vision_config)
        self.multi_modal_projector = LlavaMultiModalProjector(self.config)

    @classmethod
    def from_config(cls, config: LlavaConfig) -> "Llava":
        return cls(config)

    def get_config(self) -> LlavaConfig:
        return self.config

    def reset_parameters(self):
        self.langauage_model.reset_parameters()
        self.vision_tower.reset_parameters()
        self.multi_modal_projector.reset_parameters()

    def post_init(self):
        self.language_model.post_init()

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        _, image_features = self.vision_tower(
            pixel_values, output_hidden_states=True, **kwargs
        )
        if isinstance(self.config.vision_feature_layer, int):
            selected_image_feature = image_features[self.config.vision_feature_layer]
            if self.config.vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [
                image_features[layer_idx]
                for layer_idx in self.config.vision_feature_layer
            ]
            if self.config.vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]            
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features


    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        position_ids=None,
        past_key_value_states=None,
        inputs_embeds=None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        pdb.set_trace()
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids and inputs_embeds can't both be None")

        # input_embeds supersedes input_ids
        if inputs_embeds is None:
            inputs_embeds = self.language_model.base_model.embedding(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
            #TODO: image_features value doesn't match HF but shape does

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(
                -1
            )
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            image_features = image_features.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        #pdb.set_trace()
        outputs = self.language_model(
            inputs_embeds,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            use_cache=use_cache,
            is_input_embedded=True,
            **attn_kwargs,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        iteration,
        input_ids,
        kwargs,
    ):
        if kwargs["use_cache"] and iteration > 0:
            kwargs["pixel_values"] = None
            kwargs["image_sizes"] = None
        return input_ids, kwargs


_pixtral_12b_config = LlavaConfig()

_architecture_name = "llava"


def _llava_factory_factory(config):
    def factory(**kwargs):
        return Llava(config, **kwargs)

    return factory


models.register_model(
    _architecture_name,
    "pixtral_12b",
    _llava_factory_factory(_pixtral_12b_config),
)


def _weight_fusion(
    input_sd: Mapping, model_config: Optional[LlavaConfig] = None, **kwargs
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
        (r"attention\.k_proj", "attn.in_proj.key"),
        (r"attention\.v_proj", "attn.in_proj.value"),
        (r"attention\.q_proj", "attn.in_proj.query"),
        (r"attention\.o_proj", "attn.dense"),
        (r"feed_forward\.gate_proj", "feed_forward.wg"),
        (r"feed_forward\.up_proj", "feed_forward.w1"),
        (r"feed_forward\.down_proj", "feed_forward.w2"),
        # language
        (r"language_model\.lm_head\.weight", "language_model.head.weight"),
        (
            r"language_model.model.embed_tokens.weight",
            "language_model.base_model.embedding.weight",
        ),
        (r"language_model.model.norm", "language_model.base_model.dec_norm"),
        (r"language_model.model.layers", "language_model.base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
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


def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    if "int8" in linear_type:
        # quantize_weight is fms-model-optimizer identifier of weight clip values
        return ["weight", "bias", "quantize_weight"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


#TODO: might need _hf_to_fms_rope for pixtral as well
#TODO: combine both _hf_to_fms_rope_xxx_model() into one

def _hf_to_fms_rope_vision_model(
    input_sd: Mapping[str, Any],
    model_config=None,
    **kwargs,
) -> Mapping[str, Any]:
    new_sd = {}
    if model_config:
        model_config = model_config.vision_config

    if model_config:
        head_size = model_config.hidden_size // model_config.nheads
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
        f"vision_tower.transformer.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})"
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


def _hf_to_fms_rope_language_model(
    input_sd: Mapping[str, Any],
    model_config=None,
    **kwargs,
) -> Mapping[str, Any]:
    new_sd = {}
    if model_config:
        model_config = model_config.text_config

    if model_config:
        #head_size = model_config.emb_dim // model_config.nheads
        head_size = model_config.head_dim
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
    _architecture_name, "hf_to_fms_rope_language_model", _hf_to_fms_rope_language_model
)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope_vision_model", _hf_to_fms_rope_vision_model
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    #["hf_to_fms_names"],
    ["hf_to_fms_names", "hf_to_fms_rope_language_model", "hf_to_fms_rope_vision_model"],
    #["hf_to_fms_names", "hf_to_fms_rope", "weight_fusion"],
)
