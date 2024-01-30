import math
import re
from dataclasses import dataclass
from typing import Mapping, Optional

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
from fms.modules.feedforward import MOEFeedForward
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.config import ModelConfig
from fms.utils.serialization import FusableWeightsMissingError


@dataclass
class MixtralConfig(ModelConfig):
    src_vocab_size: int = 32_000  # can be set by tokenizer
    dim: int = 4096
    norm_eps: float = 1e-5
    nheads: int = 32
    kvheads: int = 8
    nlayers: int = 32
    pad_id: int = -1
    hidden_dim = 14336
    p_dropout: float = 0.0
    num_experts: int = 8
    top_k_experts: int = 2
    max_expected_seq_len: int = 32768
    rope_base: float = 1000000.0
    ntk_scaling: bool = False


class MixtralBlock(nn.Module):
    def __init__(self, config: MixtralConfig, rotary_emb: RotaryEmbedding):
        super(MixtralBlock, self).__init__()
        self.config = config
        emb_kq = self.config.dim // self.config.nheads
        emb_v = self.config.dim // self.config.nheads

        self.ln = LayerNormParameterized(
            self.config.dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = LayerNormParameterized(
            self.config.dim,
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
            self.config.dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=False,
            position_encoder=rotary_emb,
        )
        self.ff_sub_layer = MOEFeedForward(
            self.config.num_experts,
            self.config.top_k_experts,
            self.config.dim,
            self.config.hidden_dim,
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


class Mixtral(nn.Module):
    def __init__(
        self,
        config: Optional[MixtralConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(Mixtral, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = MixtralConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.width = self.config.dim
        self.pad_id = self.config.pad_id
        self.max_expected_seq_len = self.config.max_expected_seq_len

        shared = WordEmbedding(
            self.config.src_vocab_size,
            self.config.dim,
            padding_idx=self.config.pad_id,
            abs_pos=False,
            reversible=True,
            tie_weights=False,
            bias=False,
        )
        self.shared = self.distributed_strategy.distribute_module(shared)

        self.rot_emb = RotaryEmbedding(
            dim=self.config.dim // self.config.nheads,
            ratio=self.config.rope_base,
            ntk_scaling=self.config.ntk_scaling,
            max_seq_len=self.config.max_expected_seq_len,
        )
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
            block: nn.Module = MixtralBlock(self.config, self.rot_emb)
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        dec_norm = LayerNormParameterized(
            self.config.dim,
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

    def get_config(self) -> MixtralConfig:
        return self.config

    @classmethod
    def from_config(cls, config: MixtralConfig) -> "Mixtral":
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


# Register common Mixtral variants with the model registration API


_8x7b_config = MixtralConfig()

_architecture_name = "mixtral"


def _mixtral_factory_factory(config):
    def factory(**kwargs):
        return Mixtral(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "8x7b", _mixtral_factory_factory(_8x7b_config)
)


def _hf_sd_to_fms_sd(hf_sd: Mapping) -> Mapping:
    replacements = [
        (r"output.weight", "shared.head.weight"),
        (r"tok_embeddings.weight", "shared.emb.weight"),
        (r"^norm", "dec_norm"),
        (r"^model.layers", "layers"),
        (r"attention\.wk", "attn.key"),
        (r"attention\.wv", "attn.value"),
        (r"attention\.wq", "attn.query"),
        (r"attention\.wo", "attn.dense"),
        (r"block_sparse_moe\.w1", "ff_sub_layer.cond_ffn.w1"),
        (r"block_sparse_moe\.w2", "ff_sub_layer.cond_ffn.w2"),
        (r"block_sparse_moe\.w3", "ff_sub_layer.cond_ffn.w3"),
        (r"block_sparse_moe\.gate", "ff_sub_layer.gate"),
        (r"attention_norm", "ln"),
        (r"ffn_norm", "ff_ln"),
    ]
    new_sd = {}

    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

        if "gate" in new_name:
            weight_name = name.replace("gate", "w1")[:-7]
            if weight_name not in hf_sd:
                missing_weights = [
                    name.replace("gate", "w1")[:-7],
                    name.replace("gate", "w2")[:-7],
                    name.replace("gate", "w3")[:-7],
                ]
                raise FusableWeightsMissingError(missing_weights)

        if "w1" in new_name or "w2" in new_name or "w3" in new_name:
            gate_name = re.sub(r"w\d", "gate", name) + ".weight"
            if gate_name not in hf_sd:
                missing_weights = [
                    gate_name,
                    re.sub(r"w\d", "w1", name),
                    re.sub(r"w\d", "w2", name),
                    re.sub(r"w\d", "w3", name),
                ]
                missing_weights = [w for w in missing_weights if w != name]
                raise FusableWeightsMissingError(missing_weights)
            num_experts = hf_sd[gate_name].size(0)
            temp = new_sd[new_name]
            new_sd[new_name] = temp.reshape(
                num_experts, temp.size(0) // num_experts, temp.size(1)
            ).contiguous()

    if "gate" in new_name:
        new_sd[new_name] = new_sd[new_name].contiguous()
    return new_sd


serialization.register_adapter("mixtral", "hf", _hf_sd_to_fms_sd)
