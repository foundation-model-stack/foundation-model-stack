import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple
from typing_extensions import Unpack

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
)
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
)
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.head import LinearClassificationHead
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.linear import get_linear_type
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


"""
Qwen3 Model Implementation

Based on the Qwen3 architecture from HuggingFace:
https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

Architecture mapping from HuggingFace config.json:
- attention_bias: False -> attn_bias
- attention_dropout: 0.0 -> p_dropout
- head_dim: 128 -> head_dim
- hidden_act: "silu" -> activation_fn
- hidden_size: 1024 -> emb_dim
- intermediate_size: 3072 -> used to calculate hidden_grow_factor
- max_position_embeddings: 32768 -> max_expected_seq_len
- num_attention_heads: 16 -> nheads
- num_hidden_layers: 28 -> nlayers
- num_key_value_heads: 8 -> kvheads
- rms_norm_eps: 1e-06 -> norm_eps
- rope_theta: 1000000 -> rope_theta
- vocab_size: 151669 -> src_vocab_size
- tie_word_embeddings: true -> tie_heads
"""


@dataclass
class Qwen3Config(ModelConfig):
    src_vocab_size: int = 151_669
    emb_dim: int = 1024
    norm_eps: float = 1e-6
    nheads: int = 16
    kvheads: int = 8
    nlayers: int = 28
    pad_id: int = -1
    hidden_grow_factor: float = 3072 / 1024  # intermediate_size / hidden_size
    multiple_of: int = 256
    activation_fn: str = "swish"  # silu is same as swish
    p_dropout: float = 0.0
    max_expected_seq_len: int = 32768
    attn_bias: bool = False
    mlp_bias: bool = False
    tie_heads: bool = True
    rope_theta: float = 1000000.0
    rope_scaling: dict = field(default_factory=lambda: {})
    head_dim: int = 128
    linear_config: Optional[Mapping[str, Any]] = None
    fused_weights: bool = False


class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config, rotary_emb: RotaryEmbedding):
        super(Qwen3Block, self).__init__()
        self.config = config
        emb_kq = self.config.head_dim
        emb_v = self.config.head_dim

        self.ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )

        if self.config.kvheads == 0:
            kvheads = self.config.nheads
        else:
            kvheads = self.config.kvheads
            assert self.config.nheads % self.config.kvheads == 0

        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=self.config.attn_bias,
            position_encoder=rotary_emb,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
            apply_norm_per_head=True,
            norm_eps=self.config.norm_eps,
            head_dim=self.config.head_dim,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=self.config.mlp_bias,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x,
        *,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
        x = self.attn(
            q=x,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
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
            return (x, cache)
        else:
            return x


class Qwen3Headless(nn.Module):
    def __init__(
        self,
        config: Optional[Qwen3Config] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(Qwen3Headless, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = Qwen3Config()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.embedding = nn.Embedding(
            self.config.src_vocab_size, self.config.emb_dim, self.config.pad_id
        )

        self.rot_emb = RotaryEmbedding(
            dim=self.config.head_dim,
            scaling=self.config.rope_scaling,
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
            block: nn.Module = Qwen3Block(self.config, self.rot_emb)
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

    def get_config(self) -> Qwen3Config:
        return self.config

    @classmethod
    def from_config(cls, config: Qwen3Config) -> "Qwen3Headless":
        return cls(config)

    def reset_parameters(self):
        assert isinstance(self.embedding, torch.nn.Embedding)
        nn.init.trunc_normal_(
            self.embedding.weight, mean=0.0, std=self.config.emb_dim**-0.5
        )

        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        # Call reset_parameters for relevant sub-layers
        for m in self.modules():
            if (
                isinstance(m, MultiHeadAttention)
                or isinstance(m, GatedLinearUnit)
                or isinstance(m, LayerNormParameterized)
            ):
                m.reset_parameters()

    def validate_reset_parameters(self):
        # Verifies that the above self.reset_parameters() executed correctly.
        # This may not always be the case for distributed settings with sharded tensors,
        # such as FSDP or TP. Note that performing this check may require unsharding /
        # re-materializing the full model on a single rank to access the underlying tensors.
        tolerance = 1e-3

        def check_close(x):
            assert x.mean().abs() < tolerance
            assert x.std().sub(0.02).abs() < tolerance

        with torch.no_grad():
            for p in self.parameters():
                assert p.isnan().int().sum() == 0
                assert p.isinf().int().sum() == 0
            for m in self.modules():
                if isinstance(m, LayerNormParameterized):
                    if m.elementwise_scale:
                        assert m.weight.sum() == m.weight.numel()
                    if m.elementwise_shift:
                        assert m.bias.add(1).sum() == m.bias.numel()
                elif isinstance(m, nn.Embedding):
                    check_close(m.weight)
                elif isinstance(m, GatedLinearUnit):
                    check_close(m.w1.weight)
                    check_close(m.w2.weight)
                    check_close(m.wg.weight)
                elif isinstance(m, MultiHeadAttention):
                    if m.fused:
                        check_close(m.in_proj.qkv_fused.weight)
                    else:
                        check_close(m.in_proj.query.weight)
                        check_close(m.in_proj.key.weight)
                        check_close(m.in_proj.value.weight)
                    check_close(m.dense.weight)

    def _clean_up_rot_emb_cache(
        self,
        cached_freqs: dict[Optional[torch.device], dict[int, torch.Tensor]],
        max_seq_len_cached: dict[Optional[torch.device], int],
    ):
        # remove meta tensors from cached_freqs
        for dev in list(cached_freqs.keys()):
            for alp in list(cached_freqs[dev].keys()):
                if cached_freqs[dev][alp].device == torch.device("meta"):
                    del cached_freqs[dev][alp]
                    if len(cached_freqs[dev]) == 0:
                        del cached_freqs[dev]
                        del max_seq_len_cached[dev]

    def post_init(self):
        # This function is called in `get_model` after the model is
        # fully initalized on the correct device
        self._clean_up_rot_emb_cache(
            self.rot_emb.cached_freqs,
            self.rot_emb.max_seq_len_cached,
        )

        # init RoPE on the right device(s)
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

    def forward(
        self,
        x_in,
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
        x_in = self.embedding(x_in)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x_in,
                position_ids=position_ids,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                **attn_kwargs,
            )

            if use_cache:
                x_in, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)
            else:
                x_in = output

        dec_out = x_in
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


class Qwen3(nn.Module):
    def __init__(
        self,
        config: Optional[Qwen3Config] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(Qwen3, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = Qwen3Config()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = Qwen3Headless(self.config, self.distributed_strategy)
        self.head = LinearClassificationHead(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

    def get_config(self) -> Qwen3Config:
        return self.config

    @classmethod
    def from_config(cls, config: Qwen3Config) -> "Qwen3":
        return cls(config)

    def reset_parameters(self):
        # Call reset_parameters for relevant sub-layers
        assert isinstance(self.head, torch.nn.Linear)
        self.head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * self.config.src_vocab_size)),
        )
        self.base_model.reset_parameters()

    def validate_reset_parameters(self):
        # Verifies that the above self.reset_parameters() executed correctly.
        # This may not always be the case for distributed settings with sharded tensors,
        # such as FSDP or TP. Note that performing this check may require unsharding /
        # re-materializing the full model on a single rank to access the underlying tensors.
        tolerance = 1e-3

        def check_close(x):
            assert x.mean().abs() < tolerance
            assert x.std().sub(0.02).abs() < tolerance

        with torch.no_grad():
            for p in self.parameters():
                assert p.isnan().int().sum() == 0
                assert p.isinf().int().sum() == 0
            self.base_model.validate_reset_parameters()
            check_close(self.head.weight)

    def post_init(self):
        self.base_model.post_init()

        # if this model ties weights, they are tied here
        if self.config.tie_heads:
            # handle assignment of non-meta weights to meta parameters
            if self.head.weight.device == torch.device("meta"):
                self.head.weight = self.base_model.embedding.weight
            else:
                # For torch.compile compatibility, copy weights instead of tying them
                # This avoids graph tracing issues with shared tensors in certain backends
                with torch.no_grad():
                    self.base_model.embedding.weight.copy_(self.head.weight)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: bool = False,
        last_n_tokens: int = 0,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        output, cache = self.base_model(
            x, position_ids, past_key_value_states, use_cache, **attn_kwargs
        )

        output = self.head(output)

        if use_cache:
            return output, cache
        else:
            return output


# Register Qwen3 variants with the model registration API

# Qwen3-Embedding-0.6B configuration
_0_6b_config = Qwen3Config(
    src_vocab_size=151669,
    emb_dim=1024,
    norm_eps=1e-6,
    nheads=16,
    kvheads=8,
    nlayers=28,
    hidden_grow_factor=3072 / 1024,
    max_expected_seq_len=32768,
    rope_theta=1000000.0,
    head_dim=128,
    tie_heads=True,
)

_4b_config = Qwen3Config(
    src_vocab_size=151665,
    emb_dim=2560,
    norm_eps=1e-6,
    nheads=32,
    kvheads=8,
    nlayers=36,
    hidden_grow_factor=9728 / 2560,
    max_expected_seq_len=40960,
    rope_theta=1000000.0,
    head_dim=128,
    tie_heads=True,
)

_architecture_name = "qwen3"


def _qwen3_factory_factory(config):
    def factory(**kwargs):
        return Qwen3(config, **kwargs)

    return factory


models.register_model(_architecture_name, "0.6b", _qwen3_factory_factory(_0_6b_config))
models.register_model(_architecture_name, "4b", _qwen3_factory_factory(_4b_config))


# HuggingFace checkpoint adapter
def _hf_to_fms_names(
    hf_sd: Mapping[str, Any], model_config: Optional[Qwen3Config] = None
) -> Mapping[str, Any]:
    """
    Convert HuggingFace Qwen3 state dict to FMS format
    """
    replacements = [
        (r"^lm_head.weight", "head.weight"),
        # Qwen3 Embedding models have no "model." prefix, but the generative ones do
        (r"^model.", ""),
        (r"^norm.weight", "base_model.dec_norm.weight"),
        (r"^embed_tokens.weight", "base_model.embedding.weight"),
        (r"layers", "base_model.layers"),
        (r"self_attn.k_proj.weight", "attn.in_proj.key.weight"),
        (r"self_attn.k_norm.weight", "attn.in_proj.k_norm.weight"),
        (r"self_attn.v_proj.weight", "attn.in_proj.value.weight"),
        (r"self_attn.q_proj.weight", "attn.in_proj.query.weight"),
        (r"self_attn.q_norm.weight", "attn.in_proj.q_norm.weight"),
        (r"self_attn.o_proj.weight", "attn.dense.weight"),
        (r"mlp.gate_proj.weight", "ff_sub_layer.wg.weight"),
        (r"mlp.up_proj.weight", "ff_sub_layer.w1.weight"),
        (r"mlp.down_proj.weight", "ff_sub_layer.w2.weight"),
        (r"input_layernorm.weight", "ln.weight"),
        (r"post_attention_layernorm.weight", "ff_ln.weight"),
    ]

    new_sd = {}
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    return new_sd


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[Qwen3Config] = None, **kwargs
) -> Mapping[str, Any]:
    new_sd = {}

    for name, param in input_sd.items():
        # Some checkpoints have weights in different precisions, which can have
        # auxiliary tensors (see _get_rope_params e.g. gptq, fp8).
        # Thus, we need to get rope_params per parameter.
        linear_type_str = "torch_linear"
        if model_config and model_config.linear_config:
            linear_type_str = get_linear_type(
                model_config.linear_config,
                module_name=name,
            )
        rope_params = _get_rope_params(linear_type_str)
        trans_required_pattern = re.compile(
            f"base_model.layers.[0-9]+.attn.in_proj.(query|key|q_norm|k_norm).({'|'.join(rope_params)})$"
        )

        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters for Qwen models
        # This transpose is due to the different implementation of RoPE in
        # HF and FMS. While FMS follows the original RoPE paper
        # (https://arxiv.org/abs/2104.09864), HF has its own implementation
        # that doesn't respect the order of outputs. This is OK as long as you
        # rearrange the weights of the query and key projections, as the
        # combination projection + RoPE ends up producing the same outputs.
        # Therefore, to make FMS produce the correct order of outputs when
        # loading from an HF checkpoint, we need to undo the transformation
        # that HF does from the original Meta weights
        if bool(trans_required_pattern.match(name)) and param.numel() > 1:
            temp = param
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            assert model_config is not None and model_config.head_dim is not None
            head_size = model_config.head_dim
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # 1-dim parameters
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


def _get_rope_params(linear_type: str) -> list[str]:
    if "int8" in linear_type:
        # quantize_weight is fms-model-optimizer identifier of weight clip values
        return ["weight", "bias", "quantize_weight"]
    elif "fp8" in linear_type:
        return ["weight", "weight_scale", "input_scale", "bias"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope"],
)
