import math
import re
from dataclasses import dataclass
from typing import Any, List, Mapping, MutableMapping, Optional, Tuple

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
)
from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import MOEFeedForward
from fms.modules.head import LinearClassificationHead
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.config import ModelConfig


@dataclass
class MixtralConfig(ModelConfig):
    src_vocab_size: int = 32_000  # can be set by tokenizer
    dim: int = 4096
    norm_eps: float = 1e-5
    nheads: int = 32
    kvheads: int = 8
    nlayers: int = 32
    hidden_dim: int = 14336
    p_dropout: float = 0.0
    num_experts: int = 8
    top_k_experts: int = 2
    max_expected_seq_len: int = 32768
    rope_base: float = 1000000.0
    ntk_scaling: bool = False
    fused_weights: bool = True  # Doesn't work with False!


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


class MixtralHeadless(nn.Module):
    def __init__(
        self,
        config: Optional[MixtralConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(MixtralHeadless, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = MixtralConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.width = self.config.dim
        self.max_expected_seq_len = self.config.max_expected_seq_len

        embedding = nn.Embedding(self.config.src_vocab_size, self.config.dim)
        self.embedding = self.distributed_strategy.distribute_module(embedding)

        self.rot_emb = RotaryEmbedding(
            dim=self.config.dim // self.config.nheads,
            ratio=self.config.rope_base,
            ntk_scaling=self.config.ntk_scaling,
            max_seq_len=self.config.max_expected_seq_len,
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

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.embedding.weight, mean=0.0, std=self.config.dim**-0.5
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
                or isinstance(m, MOEFeedForward)
                or isinstance(m, LayerNormParameterized)
            ):
                m.reset_parameters()

    def post_init(self):
        # This function is called in `get_model` after the model is fully initalized in the correct device

        # init RoPE on the right device(s)
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [
                (torch.empty(0), torch.empty(0)) for _ in range(len(self.layers))
            ]

        qlen = x.size(1)
        klen = x.size(1)

        # if we are using the cache, the key length needs to be extended with the past keys length
        if use_cache and past_key_value_states[0][0].numel() > 0:
            klen += past_key_value_states[0][0].size(1)

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

        x = self.embedding(x)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x,
                mask=mask,
                position_ids=position_ids,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                is_causal_mask=is_causal_mask,
                attn_algorithm=attn_algorithm,
            )

            if use_cache:
                x, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)

            else:
                x = output

        dec_out = x
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


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

        self.base_model = MixtralHeadless(self.config, self.distributed_strategy)
        head = LinearClassificationHead(
            self.config.dim, self.config.src_vocab_size, bias=False
        )
        self.head = self.distributed_strategy.distribute_module(head)

    def get_config(self) -> MixtralConfig:
        return self.config

    @classmethod
    def from_config(cls, config: MixtralConfig) -> "Mixtral":
        return cls(config)

    def reset_parameters(self):
        # We're just going to down-scale the final prediction head to be
        # mixed-fan (inputs and gradients scale to the same inverse factors) if it isn't tied
        self.head.weight.data.normal_(
            0, 1 / math.sqrt(math.sqrt(self.config.dim * self.config.src_vocab_size))
        )

        # Call reset_parameters for relevant sub-layers
        self.base_model.reset_parameters()

    def post_init(self):
        # This function is called in `get_model` after the model is fully initalized in the correct device
        self.base_model.post_init()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        only_last_token: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        output, cache = self.base_model(
            x, mask, position_ids, past_key_value_states, use_cache, attn_algorithm
        )

        if only_last_token:
            output = output[:, -1, :]
        preds = self.head(output)

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


# Create all the pieces to generate adapters for different checkpoints
serialization.register_adapter_step(
    _architecture_name,
    "pre0.0.6_attn_unfused_to_fused",
    serialization._pre006_attn_adapter_step,
)


def _weight_fusion(
    input_sd: Mapping[str, Any], model_config: Optional[MixtralConfig] = None, **kwargs
) -> Mapping[str, Any]:
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            raise ValueError("Unfused weights unsupported on FMS Mixtral!")

    new_sd: MutableMapping[str, Any] = dict(input_sd)
    if has_fused_weights:
        for key in list(new_sd.keys()):
            if key not in new_sd:
                continue
            if "w1" in key:
                w3_weight = key.replace("w1", "w3")
                fused_name = key.replace("w1", "w13")
                new_sd[fused_name] = torch.cat([new_sd[key], new_sd[w3_weight]], dim=1)
                del new_sd[key]
                del new_sd[w3_weight]

        new_sd = dict(serialization._attn_unfused_to_fused_step(new_sd))

    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"output.weight", "head.weight"),
        (r"tok_embeddings.weight", "base_model.embedding.weight"),
        (r"^norm", "base_model.dec_norm"),
        (r"^layers", "base_model.layers"),
        (r"attention\.wk", "attn.in_proj.key"),
        (r"attention\.wv", "attn.in_proj.value"),
        (r"attention\.wq", "attn.in_proj.query"),
        (r"attention\.wo", "attn.dense"),
        (r"block_sparse_moe\.w1", "ff_sub_layer.cond_ffn.w1"),
        (r"block_sparse_moe\.w2", "ff_sub_layer.cond_ffn.w2"),
        (r"block_sparse_moe\.w3", "ff_sub_layer.cond_ffn.w3"),
        (r"block_sparse_moe\.gate", "ff_sub_layer.gate"),
        (r"attention_norm", "ln"),
        (r"ffn_norm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

        if "gate" in new_name:
            weight_name = name.replace("gate", "w1")[:-7]
            if weight_name not in input_sd:
                missing_weights = [
                    name.replace("gate", "w1")[:-7],
                    name.replace("gate", "w2")[:-7],
                    name.replace("gate", "w3")[:-7],
                ]
                raise ValueError(f"Missing {missing_weights}")

        if "w1" in new_name or "w2" in new_name or "w3" in new_name:
            gate_name = re.sub(r"w\d", "gate", name) + ".weight"
            if gate_name not in input_sd:
                missing_weights = [
                    gate_name,
                    re.sub(r"w\d", "w1", name),
                    re.sub(r"w\d", "w2", name),
                    re.sub(r"w\d", "w3", name),
                ]
                missing_weights = [w for w in missing_weights if w != name]
                raise ValueError(f"Missing {missing_weights}")
            num_experts = input_sd[gate_name].size(0)
            temp = new_sd[new_name]
            new_sd[new_name] = temp.reshape(
                num_experts, temp.size(0) // num_experts, temp.size(1)
            ).contiguous()

    for key in list(new_sd.keys()):
        if key not in new_sd:
            continue
        if "gate" in key:
            new_sd[key] = new_sd[key].contiguous()
        if "w2" in key:
            new_sd[key] = new_sd[key].transpose(1, 2).contiguous()

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
