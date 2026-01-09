import logging
import re
from typing import Any, Mapping, Optional

import torch.nn as nn

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils import serialization
from fms.models.granite import (
    Granite,
    GraniteBlock,
    GraniteConfig,
    GraniteHeadless,
    _hf_gptq_granite_check,
    _hf_to_fms_rope,
    _weight_fusion,
)
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils.activation import str_to_activation

logger = logging.getLogger(__name__)


class GraniteMoeHybridBlock(GraniteBlock):
    def __init__(self, config: GraniteConfig, rotary_emb: RotaryEmbedding):
        super(GraniteMoeHybridBlock, self).__init__(config, rotary_emb)

        # Override ff_sub_layer with granite-4-dense specific settings
        # as it comes with fused weights
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=False,  # Granite 4 does not define MLP bias
            fused=True,  # Granite 4 comes with fused weights
            linear_config=self.config.linear_config,
        )


class GraniteMoeHybridHeadless(GraniteHeadless):
    def __init__(
        self,
        config: Optional[GraniteConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(GraniteMoeHybridHeadless, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = GraniteConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.width = self.config.emb_dim
        self.pad_id = self.config.pad_id
        self.max_expected_seq_len = self.config.max_expected_seq_len

        self.embedding = nn.Embedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            padding_idx=self.config.pad_id,
        )

        rope_scaling = {"rope_type": "ntk" if self.config.ntk_scaling else "regular"}

        self.rot_emb = RotaryEmbedding(
            dim=self.config.head_dim,
            scaling=rope_scaling,
            max_seq_len=self.config.max_expected_seq_len,
            ratio=self.config.rope_theta,
        )

        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = GraniteMoeHybridBlock(self.config, self.rot_emb)
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        dec_norm = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.dec_norm = self.distributed_strategy.distribute_module(
            dec_norm, final_layers=True
        )

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)


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
        super(GraniteMoeHybrid, self).__init__()

        if config is not None:
            self.config = config
        else:
            self.config = GraniteConfig()

        self.config = self.config.updated(**kwargs)

        self.distributed_strategy = distributed_strategy

        self.base_model = GraniteMoeHybridHeadless(
            self.config, self.distributed_strategy
        )
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

    @classmethod
    def from_config(cls, config: GraniteConfig) -> "GraniteMoeHybrid":
        return cls(config)


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
        # Following layers are different in granite-v4-dense from granite-3
        (r"shared_mlp\.input_linear", "ff_sub_layer.wg1_fused"),
        (r"shared_mlp\.output_linear", "ff_sub_layer.w2"),
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
