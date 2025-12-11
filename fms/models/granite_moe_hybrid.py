import logging
import re
from typing import Any, Mapping

from fms import models
from fms.utils import serialization
from fms.models.granite import (
    _8b_config,
    _granite_factory_factory,
    _hf_gptq_granite_check,
    _hf_to_fms_rope,
    _weight_fusion)


logger = logging.getLogger(__name__)

_architecture_name = "granite_v4"


def _hf_to_fms_names_v4(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"^lm_head.weight", "head.weight"),
        (r"^model.embed_tokens.weight", "base_model.embedding.weight"),
        (r"^model.norm", "base_model.dec_norm"),
        (r"^model.layers", "base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"shared_mlp\.output_linear", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for new_name, param in input_sd.items():
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)

        if 'shared_mlp.input_linear.weight' in new_name:
            gate_name = new_name.replace("shared_mlp.input_linear", "ff_sub_layer.wg")
            up_proj_name = new_name.replace("shared_mlp.input_linear", "ff_sub_layer.w1")
            gate_proj_weight, up_proj_weight = param.chunk(2, dim=0)
            new_sd[gate_name] = gate_proj_weight
            new_sd[up_proj_name] = up_proj_weight

        else:
            new_sd[new_name] = param
    return new_sd

models.register_model(_architecture_name, "8b", _granite_factory_factory(_8b_config))

serialization.register_adapter_step(
    _architecture_name, "hf_gptq_fusion_check", _hf_gptq_granite_check
)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names_v4", _hf_to_fms_names_v4
)

serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


serialization.register_adapter(
    _architecture_name,
    "hf",
    [
        "hf_to_fms_names_v4",
        "hf_to_fms_rope",
        "hf_gptq_fusion_check",
        "weight_fusion",
    ],
)