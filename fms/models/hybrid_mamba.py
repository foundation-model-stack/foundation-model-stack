import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn

from fms import distributed, models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
    TensorParallelStrategy,
    UniformModelParallelStrategy,
)
from fms.models.llama import LLaMABlock, LLaMAConfig
from fms.modules.attention import MultiHeadAttention
from fms.modules.embedding import WordEmbedding
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.tokenizers import _has_hf, get_tokenizer

from mamba_ssm.modules.mamba2 import Mamba2


@dataclass
class HybridConfig(LLaMAConfig):
    # extends the llama config to include mamba-2 parameters
    use_mamba: bool = True
    mamba_d_model: int = 4096  # same as emb_dim
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_conv_init = None
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_d_ssm = None
    mamba_ngroups: int = 1
    mamba_A_init_range: tuple = (1, 16)
    mamba_D_has_hdim: bool = False
    mamba_rmsnorm: bool = True
    mamba_norm_before_gate: bool = False
    mamba_dt_min: float = 0.001
    mamba_dt_max: float = 0.1
    mamba_dt_init_floor: float = 1e-4
    mamba_dt_limit: tuple = (0.0, float("inf"))
    mamba_bias: bool = False
    mamba_conv_bias: bool = True
    mamba_chunk_size: int = 256
    mamba_use_mem_eff_path: bool = True


# creating a hybrid block that can switch between attention and mamba-2 blocks
# when training set dtype to torch.float32
class HybridBlock(nn.Module):
    def __init__(self, config: HybridConfig, rotary_emb: RotaryEmbedding):
        super(HybridBlock, self).__init__()
        self.use_mamba = config.use_mamba
        self.config = config

        if self.use_mamba:
            # mamba-2 block
            self.mamba_block = Mamba2(
                d_model=config.mamba_d_model,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                conv_init=config.mamba_conv_init,
                expand=config.mamba_expand,
                headdim=config.mamba_headdim,
                d_ssm=config.mamba_d_ssm,
                ngroups=config.mamba_ngroups,
                A_init_range=config.mamba_A_init_range,
                D_has_hdim=config.mamba_D_has_hdim,
                rmsnorm=config.mamba_rmsnorm,
                norm_before_gate=config.mamba_norm_before_gate,
                dt_min=config.mamba_dt_min,
                dt_max=config.mamba_dt_max,
                dt_init_floor=config.mamba_dt_init_floor,
                dt_limit=config.mamba_dt_limit,
                bias=config.mamba_bias,
                conv_bias=config.mamba_conv_bias,
                chunk_size=config.mamba_chunk_size,
                use_mem_eff_path=config.mamba_use_mem_eff_path,
            )
        else:
            self.llama_block = LLaMABlock(config, rotary_emb)

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
        if self.use_mamba:
            return self.mamba_block(x), None
        else:
            return self.llama_block(
                x,
                mask=mask,
                position_ids=position_ids,
                past_key_value_state=past_key_value_state,
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                attn_algorithm=attn_algorithm,
            )


# modifying the llama model to use the HybridBlock
class HybridLLaMA(nn.Module):
    def __init__(
        self,
        config: Optional[HybridConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(HybridLLaMA, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = HybridConfig()
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
        )
        # RoPE init
        if isinstance(self.distributed_strategy, UniformModelParallelStrategy):
            for dev_idx in set(self.distributed_strategy.layer_to_device):
                self.rot_emb.compute_freqs_cis(
                    torch.device("cuda", dev_idx), self.config.max_expected_seq_len
                )
        else:
            self.rot_emb.compute_freqs_cis(
                self.shared.emb.weight.device, self.config.max_expected_seq_len
            )

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = HybridBlock(self.config, self.rot_emb)
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

    def get_config(self) -> HybridConfig:
        return self.config

    @classmethod
    def from_config(cls, config: HybridConfig) -> "HybridLLaMA":
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

    def _helper(
        self,
        x_in,
        mask=None,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        attn_algorithm=None,
    ):
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        qlen = x_in.size(1)
        klen = x_in.size(1)

        if use_cache and past_key_value_states[0] is not None:
            klen += past_key_value_states[0][0].size(-2)

        if mask is None:
            if use_cache and klen != 1 and qlen == 1:
                is_causal_mask = False
            else:
                is_causal_mask = True
        else:
            is_causal_mask = False

        x_in = self.shared(x_in)

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
                x_in = output[0]

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
_micro_char_config = HybridConfig(
    emb_dim=192, nheads=4, nlayers=5, max_expected_seq_len=1024, src_vocab_size=256
)

_7b_config = HybridConfig()

_architecture_name = "hybrid_mamba"


def hybrid_mamba_factory_factory(config):
    def factory(**kwargs):
        return HybridLLaMA(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "micro", hybrid_mamba_factory_factory(_micro_char_config)
)
models.register_model(
    _architecture_name, "7b", hybrid_mamba_factory_factory(_7b_config)
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
