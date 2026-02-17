import functools
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple
from typing_extensions import Unpack

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    get_attention_type,
)
from fms.modules.feedforward import FeedForwardBlock
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig
from fms.utils.headless import gather_outputs


@dataclass
class GPTBigCodeConfig(ModelConfig):
    # This param default is based on https://huggingface.co/bigcode/gpt_bigcode-santacoder
    src_vocab_size: int = 49157
    # This param default is based on https://huggingface.co/bigcode/gpt_bigcode-santacoder
    emb_dim: int = 2048
    nheads: int = 12
    nlayers: int = 12
    pad_id: int = 0
    max_expected_seq_len: int = 512
    hidden_grow_factor: float = 4.0
    activation_fn: str = "gelu-tanh"
    p_dropout: float = 0.0
    emb_dropout: float = 0.0
    multiquery_attn: bool = True
    ln_eps: float = 1e-5
    linear_config: Optional[Mapping[str, Any]] = (
        None  # pass as {"linear_type": str, <other kwargs>}
    )
    fused_weights: bool = True
    tie_heads: bool = True


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        super().__init__()
        self.config = config

        self.ln = nn.LayerNorm(self.config.emb_dim, self.config.ln_eps)
        self.ff_ln = nn.LayerNorm(self.config.emb_dim, self.config.ln_eps)

        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            self.config.emb_dim // self.config.nheads,
            self.config.emb_dim // self.config.nheads,
            self.config.nheads,
            kvheads=1 if self.config.multiquery_attn else self.config.nheads,
            p_dropout=self.config.p_dropout,
            use_bias=True,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )

        self.ff_sub_layer = FeedForwardBlock(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=True,
            linear_config=self.config.linear_config,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        self_attn_past_key_value = past_key_value_state

        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
        # self attention
        x = self.attn(
            q=x,
            position_ids=position_ids,
            past_key_value_state=self_attn_past_key_value,
            use_cache=use_cache,
            **attn_kwargs,
        )

        cache = None
        if use_cache:
            x, cache = x
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # residual connection
        x = x + residual

        # then we do FF and Add&Norm
        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # another residual
        x = x + residual

        if use_cache:
            return x, cache
        else:
            return x


class GPTBigCodeHeadless(nn.Module):
    def __init__(
        self, config: GPTBigCodeConfig, distributed_strategy: DistributedStrategy
    ):
        super().__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        layers = []
        for i in range(self.config.nlayers):
            block = GPTBigCodeBlock(self.config)
            block_module = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block_module)
        self.layers = nn.ModuleList(layers)

        self.embedding = nn.Embedding(self.config.src_vocab_size, self.config.emb_dim)
        self.position_embedding = nn.Embedding(
            self.config.max_expected_seq_len, self.config.emb_dim
        )

        self.dec_norm = self.distributed_strategy.distribute_module(
            nn.LayerNorm(self.config.emb_dim, eps=self.config.ln_eps), final_layers=True
        )

        if self.config.emb_dropout:
            self.emb_dropout = nn.Dropout(self.config.emb_dropout)

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def _compute_position_ids(
        self,
        is_pad: torch.Tensor,
        use_cache: bool,
        past_key_value_states: Optional[
            List[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]
        ] = None,
    ):
        """compute the position ids if the use happened not to give any"""
        position_ids = ((~is_pad).cumsum(1) - 1).clamp(min=0)

        # Compute position_ids based on cache config
        if (
            use_cache
            and past_key_value_states is not None
            and past_key_value_states[0] is not None
        ):
            position_ids += past_key_value_states[0][0].size(-2)

        return position_ids

    def forward(
        self,
        x: torch.LongTensor,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len

        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        x_emb = self.embedding(x)

        # if pad_id exists
        #   is_pad will be a BoolTensor
        #   otherwise pad_id will not be taken into account
        if self.config.pad_id is None:
            is_pad = torch.zeros_like(x, dtype=bool, device=x.device)
        else:
            is_pad = x == self.config.pad_id

        if position_ids is None:
            position_ids = self._compute_position_ids(
                is_pad, use_cache, past_key_value_states
            )

        # look up position embeddings
        position_out = self.position_embedding(position_ids)

        # zero out the associated position embeddings
        if self.config.pad_id is not None:
            position_out = position_out.mul(~is_pad.unsqueeze(-1))

        # perform absolute position embedding
        x = x_emb + position_out

        # apply dropout to embeddings
        if self.config.emb_dropout:
            x = self.emb_dropout(x)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x,
                position_ids=position_ids,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                **attn_kwargs,
            )

            if use_cache:
                x, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)

            else:
                x = output

        dec_out = self.dec_norm(x)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


# Implements the decoder-only GPTBigCodeModel
class GPTBigCode(nn.Module):
    def __init__(
        self,
        config: Optional[GPTBigCodeConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(GPTBigCode, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = GPTBigCodeConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = GPTBigCodeHeadless(self.config, self.distributed_strategy)
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

    @classmethod
    def from_config(cls, config: GPTBigCodeConfig) -> "GPTBigCode":
        return cls(config)

    def get_config(self) -> GPTBigCodeConfig:
        return self.config

    def reset_parameters(self):
        # Do not re-initialize head, as weights are tied
        for m in self.modules():
            if (
                isinstance(m, MultiHeadAttention)
                or isinstance(m, FeedForwardBlock)
                or isinstance(m, nn.LayerNorm)
            ):
                m.reset_parameters()
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(
                    m.weight,
                    mean=0.0,
                    std=self.config.emb_dim**-0.5,
                )

    def post_init(self):
        # This function is called in `get_model` after the model is fully initalized in the correct device

        # if this model ties weights, they are tied here
        if self.config.tie_heads:
            # handle assignment of non-meta weights to meta parameters
            if self.head.weight.device == torch.device("meta"):
                self.head.weight = self.base_model.embedding.weight
            else:
                self.base_model.embedding.weight = self.head.weight

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: bool = False,
        last_n_tokens: int = 0,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        get_attention_type(**attn_kwargs)["validate_attn_kwargs"](
            input_ids=x,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            **attn_kwargs,
        )

        output, cache = self.base_model(
            x,
            position_ids,
            past_key_value_states,
            use_cache,
            **attn_kwargs,
        )

        output = gather_outputs(output, last_n_tokens, **attn_kwargs)
        preds = self.head(output)

        if use_cache:
            return preds, cache
        else:
            return preds


_micro_char_config = GPTBigCodeConfig(
    emb_dim=192, nheads=4, nlayers=5, max_expected_seq_len=1024, src_vocab_size=256
)

# Register common GPT Bigcode variants with the model registration API
_santacoder_config = GPTBigCodeConfig(
    src_vocab_size=49280,
    emb_dim=2048,
    nheads=16,
    nlayers=24,
    pad_id=-1,
    max_expected_seq_len=2048,
    p_dropout=0.1,
    emb_dropout=0.1,
)

_3b_config = GPTBigCodeConfig(
    src_vocab_size=49152,
    emb_dim=3072,
    nheads=32,
    nlayers=32,
    pad_id=0,
    max_expected_seq_len=2048,
    hidden_grow_factor=4.0,
    activation_fn="gelu",
    multiquery_attn=True,
    ln_eps=1e-5,
)

# https://www.ibm.com/docs/en/cloud-paks/cp-data/4.8.x?topic=models-granite-13b-instruct-v2-model-card
_13b_config = GPTBigCodeConfig(
    src_vocab_size=50304,
    emb_dim=5632,
    nheads=44,
    nlayers=40,
    pad_id=50280,
    max_expected_seq_len=8192,
    hidden_grow_factor=4.0,
    p_dropout=0.1,
    emb_dropout=0.1,
    ln_eps=1e-5,
)

#  Config verified with IBM internal repo
_20b_config = GPTBigCodeConfig(
    src_vocab_size=49152,
    emb_dim=6144,
    nheads=48,
    nlayers=52,
    pad_id=0,
    max_expected_seq_len=8192,
    hidden_grow_factor=4.0,
    p_dropout=0.1,
    emb_dropout=0.1,
    ln_eps=1e-5,
)

_architecture_name = "gpt_bigcode"


def _gpt_bigcode_factory_factory(config):
    def factory(**kwargs):
        return GPTBigCode(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "micro", _gpt_bigcode_factory_factory(_micro_char_config)
)
models.register_model(
    _architecture_name, "santacoder", _gpt_bigcode_factory_factory(_santacoder_config)
)
models.register_model(
    _architecture_name, "ibm.3b", _gpt_bigcode_factory_factory(_3b_config)
)

models.register_model(
    _architecture_name, "ibm.13b", _gpt_bigcode_factory_factory(_13b_config)
)
models.register_model(
    _architecture_name, "ibm.20b", _gpt_bigcode_factory_factory(_20b_config)
)


# Create all the pieces to generate adapters for different checkpoints
serialization.register_adapter_step(
    _architecture_name,
    "pre0.0.6_attn_unfused_to_fused",
    serialization._pre006_attn_adapter_step,
)


def _gptq_unfuse(fused_weight, fused_weight_name: str, emb_size: int, pack_ratio: int):
    if "qweight" in fused_weight_name:
        out_feat = fused_weight.size(1)
        kv_out_feat = (out_feat - emb_size) // 2
        return (
            fused_weight[:, :emb_size],
            fused_weight[:, emb_size : emb_size + kv_out_feat],
            fused_weight[:, -kv_out_feat:],
        )
    elif "qzeros" in fused_weight_name:
        out_feat_packed = fused_weight.size(1)
        query_out_feat_packed = emb_size // pack_ratio
        kv_out_feat_packed = (out_feat_packed - query_out_feat_packed) // 2
        return (
            fused_weight[:, :query_out_feat_packed],
            fused_weight[
                :, query_out_feat_packed : query_out_feat_packed + kv_out_feat_packed
            ],
            fused_weight[:, -kv_out_feat_packed:],
        )
    elif "scales" in fused_weight_name:
        out_feat = fused_weight.size(1)
        kv_out_feat = (out_feat - emb_size) // 2
        return (
            fused_weight[:, :emb_size],
            fused_weight[:, emb_size : emb_size + kv_out_feat],
            fused_weight[:, -kv_out_feat:],
        )
    elif "bias" in fused_weight_name:
        out_feat = fused_weight.size()[0]
        kv_out_feat = (out_feat - emb_size) // 2
        return (
            fused_weight[:emb_size],
            fused_weight[emb_size : emb_size + kv_out_feat],
            fused_weight[-kv_out_feat:],
        )
    elif "g_idx" in fused_weight_name:
        return fused_weight, fused_weight, fused_weight


def _torch_unfuse(fused_weight, fused_weight_name: str, emb_size: int):
    out_feat = fused_weight.size(0)
    kv_out_feat = (out_feat - emb_size) // 2
    return (
        fused_weight[:emb_size],
        fused_weight[emb_size : emb_size + kv_out_feat],
        fused_weight[-kv_out_feat:],
    )


def _weight_fusion(
    input_sd: Mapping[str, Any],
    model_config: Optional[GPTBigCodeConfig] = None,
    **kwargs,
) -> Mapping[str, Any]:
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if not has_fused_weights:
        if model_config:
            emb_size = model_config.emb_dim
            is_gptq = (
                model_config.linear_config is not None
                and "gptq" in model_config.linear_config["linear_type"]
            )
        else:
            raise ValueError(
                "Missing model_config for the weight unfusion adapter not supported"
            )

        if is_gptq:
            pack_ratio = 8  # assume 4 bits => packing/compression ratio = 32 / 4
            unfuse_op = functools.partial(
                _gptq_unfuse, emb_size=emb_size, pack_ratio=pack_ratio
            )
        else:
            unfuse_op = functools.partial(_torch_unfuse, emb_size=emb_size)

        unfused_sd = {}
        for name, param in input_sd.items():
            if "attn.in_proj.qkv_fused" in name:
                query_param_name = name.replace("qkv_fused", "query")
                key_param_name = name.replace("qkv_fused", "key")
                value_param_name = name.replace("qkv_fused", "value")

                qw, kw, vw = unfuse_op(param, name)
                unfused_sd[query_param_name] = qw
                unfused_sd[key_param_name] = kw
                unfused_sd[value_param_name] = vw
            else:
                unfused_sd[name] = param

        return unfused_sd
    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


def _hf_to_fms_names(hf_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    import re

    replacements = [
        ("lm_head.weight", "head.weight"),
        (r"^transformer.wte.weight", "base_model.embedding.weight"),
        (r"^transformer.wpe.weight", "base_model.position_embedding.weight"),
        (r"^transformer.ln_f", "base_model.dec_norm"),
        (r"^transformer.h", "base_model.layers"),
        # need to do kqv manually
        (r"attn\.c_attn", "attn.in_proj.qkv_fused"),
        (r"attn\.c_proj", "attn.dense"),
        (r"mlp\.c_fc", "ff_sub_layer.w1"),
        (r"mlp\.c_proj", "ff_sub_layer.w2"),
        (r"ln_1", "ln"),
        (r"ln_2", "ff_ln"),
    ]

    new_sd = {}
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)

        new_sd[new_name] = param

    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)


serialization.register_adapter(
    _architecture_name, "hf", ["hf_to_fms_names", "weight_fusion"]
)
serialization.register_adapter(
    _architecture_name,
    "fms.pre0.0.6",
    ["pre0.0.6_attn_unfused_to_fused", "weight_fusion"],
)
