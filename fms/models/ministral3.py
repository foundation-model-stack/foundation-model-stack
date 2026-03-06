import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple
from typing_extensions import Unpack

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
)

from fms.modules.attention import AttentionKwargs

from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig

from fms.modules.layernorm import LayerNormParameterized
from fms.models.mistral import Mistral
from fms.models.mistral3 import Mistral3
from fms.models.pixtral_vision import PixtralVisionConfig, PixtralVisionModel

logger = logging.getLogger(__name__)


_architecture_name = "ministral3"

@dataclass
class Ministral3TextConfig(ModelConfig):
    src_vocab_size: int = 131072
    nheads: int = 32
    nlayers: int = 40
    hidden_grow_factor: float = 16384 / 5120  # intermediate_size / hidden_size:emb_dim
    multiple_of: int = 256  # borrowed from llama
    tie_heads: bool = False
    p_dropout: float = 0.0
    activation_fn: str = "silu"
    emb_dim: int = 5120
    head_dim: int = 128  # getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    max_expected_seq_len: int = 262144
    kvheads: int = 8
    norm_eps: float = 1e-5
    sliding_window: int = 4000 # null for ministral3 in the model itself
    rope_parameters: Dict = field(default_factory=dict)
    fused_weights: bool = True  # FMS Specific -- For CPU/GPU = T, AIU = F
    pad_id: int = -1  # borrowed from granite, we do need it
    linear_config: Optional[Mapping[str, Any]] = None  # To support quantization



@dataclass
class Ministral3Config(ModelConfig):
    """
    Composite configuration for the FMS Ministral3 multimodal model.

    This wraps a Ministral3TextConfig (text) config for Ministral3 & Pixtral vision encoder.
    The current defaults correspond to Ministral3 14B, 8B, i.e.,
    https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512
    """

    text_config: Ministral3TextConfig = field(default_factory=Ministral3TextConfig)
    vision_config: PixtralVisionConfig = field(default_factory=PixtralVisionConfig)
    projector_hidden_act: str = "gelu"
    multimodal_projector_bias: bool = False
    spatial_merge_size: int = 2
    image_token_index: int = 10
    vision_feature_layer: int | list[int] = -1
    ### FMS Specific
    fused_weights: bool = True  # True For CPU/GPU = T, False for AIU

_14b_config = Ministral3Config()


# =============== Modeling ======================

class Ministral3(Mistral3):
    pass



# =============== Registration ==================

def _ministral3_factory_factory(config):
    def factory(**kwargs):
        return Mistral3(config, **kwargs)

    return factory


models.register_model(_architecture_name, "14b", _ministral3_factory_factory(_14b_config))


# =============== Serialization ==================


def _weight_fusion(
    input_sd: Mapping[str, Any], model_config: Optional[Ministral3Config] = None, **kwargs
) -> Mapping[str, Any]:
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


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = replacements = [
        # Language Model
        (r"^language_model.lm_head.weight", "language_model.head.weight"),
        (
            r"^language_model.model.embed_tokens.weight",
            "language_model.base_model.embedding.weight",
        ),
        (r"^language_model.model.norm", "language_model.base_model.dec_norm"),
        (r"^language_model.model.layers", "language_model.base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
        # Vision Model
        (r"feed_forward\.gate_proj", "ff_sub_layer.wg"),
        (r"feed_forward\.up_proj", "ff_sub_layer.w1"),
        (r"feed_forward\.down_proj", "ff_sub_layer.w2"),
        (r"attention\.k_proj", "attn.in_proj.key"),
        (r"attention\.v_proj", "attn.in_proj.value"),
        (r"attention\.q_proj", "attn.in_proj.query"),
        (r"attention\.o_proj", "attn.dense"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[Ministral3Config] = None, **kwargs
) -> Mapping[str, Any]:
    new_sd = {}

    if model_config is None:
        # It Fall back to values for Ministral3; ModelConfig should really not be
        # optional here though, as setting the wrong head dimensions can cause a
        # lot of confusion.
        lm_head_dim = 128
        vision_head_dim = 64
        logger.warning("Missing model_config, assuming default text/vision head sizes")
    else:
        text_config = model_config.text_config
        vision_config = model_config.vision_config
        lm_head_dim = text_config.head_dim
        vision_head_dim = vision_config.hidden_size // vision_config.nheads

    # TODO: Update this if we ever need gptq for this model arch,
    # this assusmes torchj linear layers.
    rope_params = ["weight", "bias"]
    # Match on either the language model or vision tower attn qk
    trans_required_pattern = re.compile(
        "|".join(
            [
                f"language_model.base_model.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})",
                f"vision_tower.transformer.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})",
            ]
        )
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
        # that HF does from the original Meta weights:
        if bool(trans_required_pattern.search(name)):
            head_dim = lm_head_dim if "language" in name else vision_head_dim
            temp = param
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_dim

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # bias
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "weight_fusion"],
)
