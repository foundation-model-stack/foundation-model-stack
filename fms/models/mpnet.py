# Copyright 2018 The HuggingFace Inc. team, Microsoft Corporation.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MPNet model."""

import math
import re
from typing import Optional, Unpack, Any
from collections.abc import Mapping
import logging
from dataclasses import dataclass

import torch
from torch import nn

from fms.utils.config import ModelConfig
from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    get_attention_type,
)
from fms.modules.feedforward import FeedForwardBlock
from fms.utils.activation import str_to_activation
from fms.utils import serialization

logger = logging.getLogger(__name__)


@dataclass
class MpnetConfig(ModelConfig):
    src_vocab_size: int = 30_527
    emb_dim: int = 768
    nlayers: int = 12
    nheads: int = 12
    intermediate_size: int = 3072
    activation_fn: str = "gelu"
    hidden_dropout_prob: float = 0.1
    p_dropout: float = 0.1
    max_expected_seq_len: int = 512
    initializer_range: float = 0.02
    multiquery_attn: bool = False
    layer_norm_eps: float = 1e-12
    hidden_grow_factor: float = 4.0
    relative_attention_num_buckets: int = 32
    tie_heads: bool = False
    pad_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    linear_config: Optional[Mapping[str, Any]] = None
    fused_weights: bool = True


class MpnetBlock(nn.Module):
    def __init__(self, config: MpnetConfig):
        super().__init__()
        self.config = config
        kvheads = self.config.nheads
        attention_head_size = int(self.config.emb_dim / self.config.nheads)
        scale_factor = 1 / math.sqrt(attention_head_size)
        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            self.config.emb_dim // self.config.nheads,
            self.config.emb_dim // self.config.nheads,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=True,
            fused=self.config.fused_weights,
            scale_factor=scale_factor,
            linear_config=self.config.linear_config,
        )
        self.ln = nn.LayerNorm(self.config.emb_dim, self.config.layer_norm_eps)
        self.ff_sub_layer = FeedForwardBlock(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.hidden_dropout_prob,
            use_bias=True,
            linear_config=self.config.linear_config,
        )
        self.ff_ln = nn.LayerNorm(self.config.emb_dim, self.config.layer_norm_eps)
        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        position_ids=None,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        residual = x
        x = self.attn(
            q=x,
            position_ids=position_ids,
            **attn_kwargs,
        )
        x = x + residual
        x = self.ln(x)
        residual = x
        x = self.ff_sub_layer(x)
        x = x + residual
        x = self.ff_ln(x)

        return x


class MpnetHeadless(nn.Module):
    def __init__(
        self,
        config: Optional[MpnetConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = MpnetConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy
        self.word_embedding = self.distributed_strategy.distribute_module(
            nn.Embedding(
                self.config.src_vocab_size,
                self.config.emb_dim,
                padding_idx=self.config.pad_id,
            )
        )
        self.position_embeddings = self.distributed_strategy.distribute_module(
            nn.Embedding(
                self.config.max_expected_seq_len,
                self.config.emb_dim,
                padding_idx=self.config.pad_id,
            )
        )

        self.enc_norm = self.distributed_strategy.distribute_module(
            nn.LayerNorm(self.config.emb_dim, eps=self.config.layer_norm_eps)
        )
        self.dropout = self.distributed_strategy.distribute_module(
            nn.Dropout(self.config.hidden_dropout_prob)
        )
        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = MpnetBlock(self.config)
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)
        self.relative_attention_bias = self.distributed_strategy.distribute_module(
            nn.Embedding(
                self.config.relative_attention_num_buckets,
                self.config.nheads,
            )
        )

    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        val_if_large = torch.min(
            val_if_large,
            torch.full(
                size=val_if_large.size(),
                fill_value=num_buckets - 1,
                dtype=val_if_large.dtype,
                layout=val_if_large.layout,
            ),
        )
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_position_bias(self, x, num_buckets=32):
        bsz, qlen, klen = x.size(0), x.size(1), x.size(1)
        device = x.device
        context_position = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long, device=device)[None, :]

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(
            relative_position, num_buckets=num_buckets
        )
        rp_bucket = rp_bucket.to(x.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        return values

    def reset_parameters(self):
        for layer in ["word_embedding", "position_embeddings"]:
            nn.init.normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
        for layer in self.layers:
            for sublayer in ["ln", "ff_ln", "attn", "ff_sub_layer"]:
                getattr(layer, sublayer).reset_parameters()
        self.enc_norm.reset_parameters()

    def forward(
        self,
        x_in,
        position_ids,
        **kwargs,
    ):
        kwargs["attn_name"] = kwargs.get("attn_name", "sdpa_bidirectional")
        inputs_embeds = self.word_embedding(x_in)

        input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = (
                torch.arange(
                    self.config.pad_id + 1,
                    seq_length + self.config.pad_id + 1,
                    dtype=torch.long,
                    device=x_in.device,
                )
                .unsqueeze(0)
                .expand(input_shape)
            )

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.enc_norm(embeddings)
        embeddings = self.dropout(embeddings)

        position_bias = self.compute_position_bias(embeddings)
        # injecting position_bias as part of sdpa attn_mask
        # FIXME for other attentions
        if kwargs.get("mask") is not None:
            attn_mask = kwargs.get("mask")
        else:
            attn_mask = None
        if attn_mask is not None:
            while len(attn_mask.size()) != 4:
                # expects bs (x nheads) x q_len x kv_len
                attn_mask = attn_mask.unsqueeze(1)
            if attn_mask.dtype == torch.bool:
                position_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                attn_mask = position_bias
            else:
                attn_mask = attn_mask + position_bias
        else:
            attn_mask = position_bias
        kwargs["mask"] = attn_mask

        x = embeddings
        for layer in self.layers:
            x = layer(x, position_ids=position_ids, **kwargs)
        return x


class Mpnet(nn.Module):
    def __init__(
        self,
        config: Optional[MpnetConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = MpnetConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy
        self.base_model = MpnetHeadless(self.config, self.distributed_strategy)
        self.den = nn.Linear(self.config.emb_dim, self.config.emb_dim)
        self.activation = nn.Tanh()

    @classmethod
    def from_config(cls, config: MpnetConfig) -> "Mpnet":
        return cls(config)

    def get_config(self) -> MpnetConfig:
        return self.config

    def reset_parameters(self):
        self.base_model.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        get_attention_type(**attn_kwargs)["validate_attn_kwargs"](
            input_ids=x, position_ids=position_ids, **attn_kwargs
        )
        if x.size()[1] > self.config.max_expected_seq_len:
            raise ValueError("input length should be<=max_position_embeddings")
        output = self.base_model(
            x,
            position_ids,
            **attn_kwargs,
        )
        first_token_tensor = output[:, 0]
        sequence_output = output
        pooled_output = self.den(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return (sequence_output, pooled_output)


_architecture_name = "mpnet"


def _mpnet_factory_factory(config):
    def factory(**kwargs):
        return Mpnet(config, **kwargs)

    return factory


_v2_config = MpnetConfig(
    src_vocab_size=30_527,
    emb_dim=768,
    nlayers=12,
    nheads=12,
    intermediate_size=3072,
    activation_fn="gelu",
    hidden_dropout_prob=0.1,
    p_dropout=0.1,
    max_expected_seq_len=512,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    relative_attention_num_buckets=32,
    pad_id=1,
    bos_token_id=0,
    eos_token_id=2,
    fused_weights=True,
)
models.register_model(_architecture_name, "v2", _mpnet_factory_factory(_v2_config))


def _weight_fusion(
    input_sd: Mapping, model_config: Optional[MpnetConfig] = None, **kwargs
):
    has_fused_weights = True
    if model_config and not model_config.fused_weights:
        has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        new_sd = serialization._attn_unfused_to_fused_step(new_sd)
    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


def _hf_to_fms_names(hf_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (
            r"embeddings.word_embeddings.weight",
            "base_model.word_embedding.weight",
        ),
        (
            r"^encoder.relative_attention_bias.weight",
            "base_model.relative_attention_bias.weight",
        ),
        (r"^pooler.dense", "den"),
        (
            r"embeddings.position_embeddings.weight",
            "base_model.position_embeddings.weight",
        ),
        (
            r"embeddings.position_ids",
            "base_model.position_ids",
        ),
        (r"embeddings.LayerNorm", "base_model.enc_norm"),
        (r"^encoder.layer", "base_model.layers"),
        (r"output\.LayerNorm", "ff_ln"),
        (r"attention\.LayerNorm", "ln"),
        (r"attention\.attn\.k", "attn.in_proj.key"),
        (r"attention\.attn\.v", "attn.in_proj.value"),
        (r"attention\.attn\.q", "attn.in_proj.query"),
        (r"attention\.attn\.o", "attn.dense"),
        (r"intermediate\.dense", "ff_sub_layer.w1"),
        (r"output\.dense", "ff_sub_layer.w2"),
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
serialization.register_adapter("mpnet", "hf", ["hf_to_fms_names", "weight_fusion"])
