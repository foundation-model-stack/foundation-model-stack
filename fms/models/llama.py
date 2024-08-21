import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Any

import torch
import torch.nn as nn

from fms import distributed, models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
    TensorParallelStrategy,
    UniformModelParallelStrategy,
)
from fms.modules.attention import MultiHeadAttention
from fms.modules.embedding import WordEmbedding
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig
from fms.utils.tokenizers import _has_hf, get_tokenizer


# params emb_dim heads layers lr
#  7B    4096    32    32     3.0E-04
# 13B    5120    40    40     3.0E-04
# 33B    6656    52    60     1.5.E-04
# 65B    8192    64    80     1.5.E-04


@dataclass
class LLaMAConfig(ModelConfig):
    src_vocab_size: int = 32_000  # can be set by tokenizer
    emb_dim: int = 4096
    norm_eps: float = 1e-5
    nheads: int = 32
    kvheads: int = 0
    nlayers: int = 32
    pad_id: int = -1
    hidden_grow_factor: float = 8 / 3
    multiple_of: int = 256
    activation_fn: str = "swish"
    p_dropout: float = 0.0
    max_expected_seq_len: int = 4096
    ntk_scaling: bool = False
    attn_bias: bool = False
    mlp_bias: bool = False
    tie_heads: bool = False
    rope_theta: float = 10_000.0
    linear_config: Optional[Mapping[str, Any]] = None


class LLaMABlock(nn.Module):
    def __init__(self, config: LLaMAConfig, rotary_emb: RotaryEmbedding):
        super(LLaMABlock, self).__init__()
        self.config = config
        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads

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
            linear_config=self.config.linear_config,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=self.config.mlp_bias,
            linear_config=self.config.linear_config,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x,
        *,
        mask=None,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        is_causal_mask=False,
        attn_algorithm=None,
    ):
        # if the cache is not empty, we need to get the kv cache for self and cross attention
        self_attn_past_key_value = past_key_value_state
        # if past_key_value_state is not None:
        #     self_attn_past_key_value = past_key_value_state[:2]
        # else:
        #     self_attn_past_key_value = None

        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
        x = self.attn(
            q=x,
            mask=mask,
            position_ids=position_ids,
            attn_algorithm=attn_algorithm,
            past_key_value_state=self_attn_past_key_value,
            use_cache=use_cache,
            is_self=True,
            is_causal_mask=is_causal_mask,
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


class LLaMA(nn.Module):
    def __init__(
        self,
        config: Optional[LLaMAConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(LLaMA, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = LLaMAConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.width = self.config.emb_dim
        self.pad_id = self.config.pad_id
        self.max_expected_seq_len = self.config.max_expected_seq_len

        shared = WordEmbedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            padding_idx=self.config.pad_id,
            abs_pos=False,
            reversible=True,
            tie_weights=self.config.tie_heads,
            bias=False,
        )
        self.shared = self.distributed_strategy.distribute_module(shared)

        self.rot_emb = RotaryEmbedding(
            dim=self.config.emb_dim // self.config.nheads,
            ntk_scaling=self.config.ntk_scaling,
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
            block: nn.Module = LLaMABlock(self.config, self.rot_emb)
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

    def get_config(self) -> LLaMAConfig:
        return self.config

    @classmethod
    def from_config(cls, config: LLaMAConfig) -> "LLaMA":
        return cls(config)

    def reset_parameters(self):
        # Call reset_parameters for relevant sub-layers
        for m in self.modules():
            if (
                isinstance(m, MultiHeadAttention)
                or isinstance(m, WordEmbedding)
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
                if isinstance(LayerNormParameterized):
                    if m.elementwise_scale:
                        assert m.weight.sum() == m.weight.numel()
                    if m.elementwise_shift:
                        assert m.bias.add(1).sum() == m.bias.numel()
                elif isinstance(WordEmbedding):
                    check_close(m.emb.weight)
                    check_close(m.head.weight)
                elif isinstance(GatedLinearUnit):
                    check_close(m.w1.weight)
                    check_close(m.w2.weight)
                    check_close(m.wg.weight)
                elif isinstance(MultiHeadAttention):
                    check_close(m.query.weight)
                    check_close(m.key.weight)
                    check_close(m.value.weight)
                    check_close(m.dense.weight)

    def post_init(self):
        # This function is called in `get_model` after the model is fully initalized in the correct device

        # if this model ties weights, so we tie here
        if self.config.tie_heads:
            # make sure you assign the non-meta weights to the meta parameters
            if self.shared.head.weight.device == torch.device("meta"):
                self.shared.head.weight = self.shared.emb.weight
            else:
                self.shared.emb.weight = self.shared.head.weight

        # init RoPE on the right device(s)
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

    def _helper(
        self,
        x_in,
        mask=None,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        attn_algorithm=None,
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        qlen = x_in.size(1)
        klen = x_in.size(1)

        # if we are using the cache, the key length needs to be extended with the past keys length
        if use_cache and past_key_value_states[0] is not None:
            klen += past_key_value_states[0][0].size(-2)

        # if mask is none, we need to specify causal mask
        if mask is None:
            # we are caching and can assume all 1s in the mask
            if use_cache and klen != 1 and qlen == 1:
                # b x h x qlen x kvlen
                is_causal_mask = False
            else:
                is_causal_mask = True
        else:
            is_causal_mask = False

        x_in = self.shared(x_in)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x_in,
                mask=mask,
                position_ids=position_ids,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                attn_algorithm=attn_algorithm,
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

    def forward(
        self,
        x,
        mask=None,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        only_last_token=False,
        attn_algorithm=None,
    ):
        output, cache = self._helper(
            x, mask, position_ids, past_key_value_states, use_cache, attn_algorithm
        )

        if only_last_token:
            output = output[:, -1, :]
        preds = self.shared(output, reverse=True)

        if use_cache:
            return preds, cache
        else:
            return preds


# Register common LLaMA variants with the model registration API

# a micro llama model to use with a char-level tokenizer
_micro_char_config = LLaMAConfig(
    emb_dim=192, nheads=4, nlayers=5, max_expected_seq_len=1024, src_vocab_size=256
)

_7b_config = LLaMAConfig()
_13b_config = LLaMAConfig(emb_dim=5120, nheads=40, nlayers=40)
# todo: add 35B config

_70b_config = LLaMAConfig(
    emb_dim=8192,
    multiple_of=4096,
    nheads=64,
    kvheads=8,
    nlayers=80,
    hidden_grow_factor=(1.3 * 8 / 3),
)

_8b_llama3_config = LLaMAConfig(
    src_vocab_size=128256,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=32,
    hidden_grow_factor=3.5,
    multiple_of=1024,
    max_expected_seq_len=8192,
    rope_theta=500_000.0,
)

# Granite configs
_granite_7b_config = LLaMAConfig(
    src_vocab_size=32008,
)

_granite_3b_code_config = LLaMAConfig(
    src_vocab_size=49152,
    emb_dim=2560,
    pad_id=0,
    hidden_grow_factor=10240 / 2560,
    multiple_of=1,
    p_dropout=0.1,
    max_expected_seq_len=2048,
    attn_bias=True,
    mlp_bias=True,
    tie_heads=True,
)

_granite_8b_code_config = LLaMAConfig(
    src_vocab_size=49152,
    emb_dim=4096,
    kvheads=8,
    nlayers=36,
    pad_id=0,
    hidden_grow_factor=14336 / 4096,
    multiple_of=1,
    p_dropout=0.1,
    max_expected_seq_len=4096,
    attn_bias=True,
    mlp_bias=True,
    tie_heads=True,
)

_architecture_name = "llama"


def _llama_factory_factory(config):
    def factory(**kwargs):
        return LLaMA(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "micro", _llama_factory_factory(_micro_char_config)
)
# Backwards compat
models.register_model(_architecture_name, "7b", _llama_factory_factory(_7b_config))
models.register_model(_architecture_name, "13b", _llama_factory_factory(_13b_config))
models.register_model(_architecture_name, "70b", _llama_factory_factory(_70b_config))

# LLama 2 family
models.register_model(_architecture_name, "2-7b", _llama_factory_factory(_7b_config))
models.register_model(_architecture_name, "2-13b", _llama_factory_factory(_13b_config))
models.register_model(_architecture_name, "2-70b", _llama_factory_factory(_70b_config))

# LLama 3 family
models.register_model(
    _architecture_name, "3-8b", _llama_factory_factory((_8b_llama3_config))
)

# Granite family
models.register_model(
    _architecture_name, "granite-7b", _llama_factory_factory((_granite_7b_config))
)
models.register_model(
    _architecture_name,
    "granite.code-3b",
    _llama_factory_factory((_granite_3b_code_config)),
)
models.register_model(
    _architecture_name,
    "granite.code-8b",
    _llama_factory_factory((_granite_8b_code_config)),
)


_convert_to_fused = lambda sd: serialization._legacy_mlp_glu_unfused_to_fused_adapter(
    serialization._legacy_attn_unfused_to_fused_adapter(sd)
)


def _rename_meta_weights_to_fms(orig_sd):
    replacements = [
        (r"^tok_embeddings", "shared.emb"),
        (r"^norm", "dec_norm"),
        (r"^output", "shared.head"),
        (r"^layers", "layers"),
        (r"\.attention\.", ".attn."),
        (r"attn\.wq", "attn.query"),
        (r"attn\.wk", "attn.key"),
        (r"attn\.wv", "attn.value"),
        (r"attn\.wo", "attn.dense"),
        (r"attention_norm", "ln"),
        (r"feed_forward\.w1", "ff_sub_layer.wg"),
        (r"feed_forward\.w2", "ff_sub_layer.w2"),
        (r"feed_forward\.w3", "ff_sub_layer.w1"),
        (r"ffn_norm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in orig_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    fused_sd = _convert_to_fused(new_sd)

    return fused_sd


def _hf_sd_to_fms_sd(hf_sd: Mapping) -> Mapping:
    replacements = [
        (r"^lm_head.weight", "shared.head.weight"),
        (r"^model.embed_tokens.weight", "shared.emb.weight"),
        (r"^model.norm", "dec_norm"),
        (r"^model.layers", "layers"),
        (r"self_attn\.k_proj", "attn.key"),
        (r"self_attn\.v_proj", "attn.value"),
        (r"self_attn\.q_proj", "attn.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}

    trans_required_pattern = re.compile("layers.[0-9]+.attn.(query|key).(weight|bias)")
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters
        # This transpose is due to the different implementation of RoPE in
        # HF and FMS. While FMS follows the original RoPE paper
        # (https://arxiv.org/abs/2104.09864), HF has its own implementation
        # that doesn't respect the order of outputs. This is OK as long as you
        # rearrange the weights of the query and key projections, as the
        # combination projection + RoPE ends up producing the same outputs.
        # Therefore, to make FMS produce the correct order of outputs when
        # loading from an HF checkpoint, we need to undo the transformation
        # that HF does from the original Meta weights:
        if bool(trans_required_pattern.match(new_name)):
            temp = new_sd[new_name]
            # nheads is used in the transformation required for hf->fms
            if temp.size(0) == 2560:
                head_size = 80  # granite 3b code
            else:
                head_size = 128  # every other Llama model in existence
            nheads = int(temp.size(0) / head_size)

            if temp.dim() == 2:  # weight
                temp_view = temp.view(nheads, 2, -1, temp.size(1))
            else:  # bias
                temp_view = temp.view(nheads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            new_sd[new_name] = temp

    fused_sd = _convert_to_fused(new_sd)

    return fused_sd


serialization.register_adapter("llama", "meta", _rename_meta_weights_to_fms)
serialization.register_adapter("llama", "hf", _hf_sd_to_fms_sd)
serialization.register_adapter("llama", "fms.pre0.0.6", _convert_to_fused)


def convert_hf_llama(hf_model: "LlamaForCausalLM") -> LLaMA:  # type: ignore
    """
    Convert a Llama huggingface model to an fms model

    Parameters
    ----------
    hf_model: LlamaForCausalLM
        a Llama Huggingface model

    Returns
    -------
    LLaMA
        an FMS LLaMA model
    """

    if not _has_hf:
        raise ImportError(
            "in order to convert huggingface weights, you need to have transformers installed"
        )

    import re

    config = LLaMAConfig(
        src_vocab_size=hf_model.config.vocab_size,
        emb_dim=hf_model.config.hidden_size,
        norm_eps=hf_model.config.rms_norm_eps,
        nheads=hf_model.config.num_attention_heads,
        nlayers=hf_model.config.num_hidden_layers,
        hidden_grow_factor=hf_model.config.intermediate_size
        / hf_model.config.hidden_size,
        multiple_of=1,  # this is set to 1 as it is encoded in the hidden dimension
        activation_fn=hf_model.config.hidden_act,
        max_expected_seq_len=hf_model.config.max_position_embeddings,
        rope_theta=hf_model.config.rope_theta,
    )
    model = LLaMA(config)
    count_parameters = lambda m: sum(p.numel() for p in m.parameters())
    assert count_parameters(model) == count_parameters(hf_model)

    hf_sd = hf_model.model.state_dict()

    replacements = [
        (r"^embed_tokens.weight", "shared.emb.weight"),
        (r"^norm", "dec_norm"),
        (r"^layers", "layers"),
        (r"self_attn\.k_proj", "attn.key"),
        (r"self_attn\.v_proj", "attn.value"),
        (r"self_attn\.q_proj", "attn.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

    model.load_state_dict(new_sd, strict=False)
    model.shared.head.weight = hf_model.lm_head.weight

    # model.rot_emb.freqs = hf_model.model.layers[0].self_attn.rotary_emb.inv_freq
    for layer in model.layers:
        q = layer.attn.query.weight.data
        q = (
            q.view(model.config.nheads, 2, -1, q.size(1))
            .transpose(1, 2)
            .reshape(*q.size())
        )
        layer.attn.query.weight.data = q

        k = layer.attn.key.weight.data
        k = (
            k.view(model.config.nheads, 2, -1, k.size(1))
            .transpose(1, 2)
            .reshape(*k.size())
        )
        layer.attn.key.weight.data = k

    return model
