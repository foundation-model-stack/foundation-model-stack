import logging
import re
from typing import Any, Mapping, Optional

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils import serialization
from fms.models.granite import (
    Granite,
    GraniteConfig,
    _hf_gptq_granite_check,
    _hf_to_fms_rope,
    _weight_fusion,
)

logger = logging.getLogger(__name__)


class GraniteMoeHybrid(Granite):
    """Granite with MoE Hybrid

    This class currently inherits from Granite to mainly support
    granite-v4-dense model, which is quite similar to granite-v3.
    GraniteMoeHybrid class will eventually support various versions of
    Granite-v4 model and we will modify this class in future accordingly.
    """

    def __init__(
        self,
        config: Optional[GraniteConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__(config, distributed_strategy, **kwargs)


_8b_config = GraniteConfig(
    src_vocab_size=100352,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=40,
    hidden_grow_factor=12800 / 4096,
    max_expected_seq_len=8192,
    rope_theta=10000000,
    pad_id=100256,
    p_dropout=0.0,  # overwriting config.json
    tie_heads=True,
    embedding_multiplier=12.0,
    logits_scaling=16.0,
    residual_multiplier=0.22,
    attention_multiplier=0.0078125,
    fused_weights=False,
)

_architecture_name = "granite_moe_hybrid"


def _granite_moe_hybrid_factory(config):
    def factory(**kwargs):
        return GraniteMoeHybrid(config, **kwargs)

    return factory


models.register_model(_architecture_name, "8b", _granite_moe_hybrid_factory(_8b_config))


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
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

        # input_linear in granite-v4-dense is fused unlike granite-3
        # below processing is done so that we let the GraniteBlock code
        # work with both fused and fused configuration and not hardcode fused weights
        # value over there.
        if "shared_mlp.input_linear.weight" in new_name:
            gate_name = new_name.replace("shared_mlp.input_linear", "ff_sub_layer.wg")
            up_proj_name = new_name.replace(
                "shared_mlp.input_linear", "ff_sub_layer.w1"
            )
            gate_proj_weight, up_proj_weight = param.chunk(2, dim=0)
            new_sd[gate_name] = gate_proj_weight
            new_sd[up_proj_name] = up_proj_weight

        else:
            new_sd[new_name] = param
    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)


serialization.register_adapter_step(
    _architecture_name,
    "weight_expansion_for_mismatched_head_dim",
    serialization._weight_expansion_for_mismatched_head_dim,  # type: ignore[arg-type]
)


serialization.register_adapter_step(
    _architecture_name, "hf_gptq_fusion_check", _hf_gptq_granite_check
)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    [
        "hf_to_fms_names",
        "hf_to_fms_rope",
        "hf_gptq_fusion_check",
        "weight_fusion",
    ],
)
