import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
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


# params emb_dim heads layers lr
#  7B    4096    32    32     3.0E-04
# 13B    5120    40    40     3.0E-04
# 33B    6656    52    60     1.5.E-04
# 65B    8192    64    80     1.5.E-04


@dataclass
class SphinxConfig(ModelConfig):
    src_vocab_size: int = 65024  # can be set by tokenizer
    emb_dim: int = 4608
    norm_eps: float = 1e-5
    nheads: int = 36
    kvheads: int = 4
    nlayers: int = 46
    pad_id: int = 2
    hidden_grow_factor: float = 8 / 3
    multiple_of: int = 1
    activation_fn: str = "swish"
    p_dropout: float = 0.1
    max_expected_seq_len: int = 4096
    ntk_scaling: bool = False


class SphinxBlock(nn.Module):
    def __init__(self, config: SphinxConfig, rotary_emb: RotaryEmbedding):
        super(SphinxBlock, self).__init__()
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
            use_bias=True,
            position_encoder=rotary_emb,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=True,
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
            k=x,
            v=x,
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


class Sphinx(nn.Module):
    def __init__(
        self,
        config: Optional[SphinxConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(Sphinx, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = SphinxConfig()
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
            tie_weights=True,
            bias=False,
        )
        self.shared = self.distributed_strategy.distribute_module(shared)

        self.rot_emb = RotaryEmbedding(
            dim=self.config.emb_dim // self.config.nheads,
            ntk_scaling=self.config.ntk_scaling,
            max_seq_len=self.config.max_expected_seq_len,
        )
        if isinstance(self.distributed_strategy, UniformModelParallelStrategy):
            for dev_idx in set(self.distributed_strategy.layer_to_device.values()):
                self.rot_emb.compute_freqs_cis(
                    torch.device("cuda", dev_idx), self.config.max_expected_seq_len
                )
        else:
            self.rot_emb.compute_freqs_cis(
                self.shared.emb.weight.device, self.config.max_expected_seq_len
            )

        self.layers = []
        for i in range(self.config.nlayers):
            block = SphinxBlock(self.config, self.rot_emb)
            block = self.distributed_strategy.distribute_layer(block, i)
            self.layers.append(block)
        self.layers = nn.ModuleList(self.layers)

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

        self.reset_params()

    def get_config(self) -> SphinxConfig:
        return self.config

    @classmethod
    def from_config(cls, config: SphinxConfig) -> "Sphinx":
        return cls(config)

    def reset_params(self):
        # Modules are self-initializing, we're just going to down-scale the final prediction head to be
        # mixed-fan (inputs and gradients scale to the same inverse factors) if it isn't tied
        self.shared.head.weight.data.normal_(
            0, 1 / math.sqrt(math.sqrt(self.width * self.shared.vocab_size))
        )

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


_1b_config = SphinxConfig(
    src_vocab_size=50304,
    emb_dim=2048,
    nheads=16,
    kvheads=4,
    nlayers=24,
    pad_id=0,
    hidden_grow_factor=5464 / 2048,
    multiple_of=1,
    max_expected_seq_len=2048,
)

_8b_config = SphinxConfig(
    src_vocab_size=65024,
    emb_dim=4608,
    nheads=36,
    kvheads=4,
    nlayers=36,
    pad_id=2,
    hidden_grow_factor=12288 / 4608,
    multiple_of=1,
    max_expected_seq_len=4096,
)

_13b_config = SphinxConfig(
    src_vocab_size=65024,
    emb_dim=5120,
    nheads=40,
    kvheads=4,
    nlayers=48,
    pad_id=2,
    hidden_grow_factor=13696 / 5120,
    multiple_of=1,
    max_expected_seq_len=4096,
)

_architecture_name = "sphinx"


def _sphinx_factory_factory(config):
    def factory(**kwargs):
        return Sphinx(config, **kwargs)

    return factory


models.register_model(_architecture_name, "1b", _sphinx_factory_factory(_1b_config))

models.register_model(_architecture_name, "8b", _sphinx_factory_factory(_8b_config))

models.register_model(_architecture_name, "13b", _sphinx_factory_factory(_13b_config))


def _megatron_sd_to_fms_sd(hf_sd: Mapping[Any, Any]) -> Mapping[Any, Any]:
    replacements = [
        # embedding
        (r"^transformer\.wte\.weight", "shared.emb.weight"),
        # layers
        (r"^transformer\.h", "layers"),
        # attn
        (r"attn\.c_proj", "attn.dense"),
        # mlp
        (r"mlp\.c_proj", "ff_sub_layer.w2"),
        # block ln
        (r"ln_1\.weight", "ln.weight"),
        (r"ln_2\.weight", "ff_ln.weight"),
        # model ln
        (r"^transformer\.ln_f\.weight", "dec_norm.weight"),
        # model head
        (r"^lm_head\.weight", "shared.head.weight"),
    ]

    is_1b = hf_sd["transformer.wte.weight"].shape[1] == 2048
    is_8b = hf_sd["transformer.wte.weight"].shape[1] == 4608
    is_13b = hf_sd["transformer.wte.weight"].shape[1] == 5120
    if is_1b:
        num_heads = 16
        num_key_value_heads = 4
        emb_dim = 2048
        n_inner = int(emb_dim * (5464 / 2048))
    elif is_8b:
        num_heads = 36
        num_key_value_heads = 4
        emb_dim = 4608
        n_inner = int(emb_dim * (12288 / 4608))
    elif is_13b:
        num_heads = 40
        num_key_value_heads = 4
        emb_dim = 5120
        n_inner = int(emb_dim * (13696 / 5120))
    else:
        raise ValueError("the state dict given is not one of Sphinx 1b, 8b, 13b")

    mlp_splits = [n_inner, n_inner]
    attn_splits = [
        (num_heads * (emb_dim // num_heads)) // num_key_value_heads,
        (num_key_value_heads * (emb_dim // num_heads)) // num_key_value_heads,
        (num_key_value_heads * (emb_dim // num_heads)) // num_key_value_heads,
    ]

    qkv_weight_pattern = re.compile("transformer.h.[0-9]+.attn.c_attn.weight")
    qkv_bias_pattern = re.compile("transformer.h.[0-9]+.attn.c_attn.bias")
    mlp_weight_pattern = re.compile("transformer.h.[0-9]+.mlp.c_fc.weight")
    mlp_bias_pattern = re.compile("transformer.h.[0-9]+.mlp.c_fc.bias")
    new_sd = {}
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

        # qkv fused
        if bool(qkv_weight_pattern.match(name)):
            new_sd.pop(new_name)
            prefix = new_name.replace("c_attn.weight", "")
            q, k, v = param.view(num_key_value_heads, -1, emb_dim).split(
                attn_splits, dim=1
            )
            q = q.reshape(-1, q.size(2))
            k = k.reshape(-1, k.size(2))
            v = v.reshape(-1, v.size(2))
            q = q.view(num_heads, 2, -1, q.size(1)).transpose(1, 2).reshape(*q.size())
            k = (
                k.view(num_key_value_heads, 2, -1, k.size(1))
                .transpose(1, 2)
                .reshape(*k.size())
            )

            new_sd[f"{prefix}query.weight"] = q
            new_sd[f"{prefix}key.weight"] = k
            new_sd[f"{prefix}value.weight"] = v
        elif bool(qkv_bias_pattern.match(name)):
            new_sd.pop(new_name)
            prefix = new_name.replace("c_attn.bias", "")
            q, k, v = param.view(num_key_value_heads, -1).split(attn_splits, dim=1)
            q = q.reshape(-1)
            k = k.reshape(-1)
            v = v.reshape(-1)
            q = q.view(num_heads, 2, -1).transpose(1, 2).reshape(*q.size())
            k = k.view(num_key_value_heads, 2, -1).transpose(1, 2).reshape(*k.size())

            new_sd[f"{prefix}query.bias"] = q
            new_sd[f"{prefix}key.bias"] = k
            new_sd[f"{prefix}value.bias"] = v
        elif bool(mlp_weight_pattern.match(name)):
            new_sd.pop(new_name)
            prefix = new_name.replace("mlp.c_fc.weight", "")
            w1, wg = param.split(mlp_splits, dim=0)
            new_sd[f"{prefix}ff_sub_layer.w1.weight"] = w1
            new_sd[f"{prefix}ff_sub_layer.wg.weight"] = wg
        elif bool(mlp_bias_pattern.match(name)):
            new_sd.pop(new_name)
            prefix = new_name.replace("mlp.c_fc.bias", "")
            w1, wg = param.split(mlp_splits, dim=0)
            new_sd[f"{prefix}ff_sub_layer.w1.bias"] = w1
            new_sd[f"{prefix}ff_sub_layer.wg.bias"] = wg

    return new_sd


serialization.register_adapter("sphinx", "megatron", _megatron_sd_to_fms_sd)
