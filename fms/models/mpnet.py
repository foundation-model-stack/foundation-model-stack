
# coding=utf-8
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
from typing import Optional, Tuple, Union, Unpack, Mapping, Any
import logging
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

#from fms.utils.activation import gelu
from fms.utils.config import ModelConfig
from fms import models
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    get_attention_type,
)
from fms.modules.layernorm import LayerNormParameterized
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
    attention_probs_dropout_prob: float = 0.1
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
    fused_weights: bool = False


class MpnetBlock(nn.Module):
    def __init__(self, config: MpnetConfig):
        super().__init__()
        self.config = config
        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads

        kvheads = self.config.nheads
        #self.ln = LayerNormParameterized(
        #    self.config.emb_dim,
        #    elementwise_scale=True,
        #    elementwise_shift=False,
        #    use_mean=False,
        #    eps=self.config.layer_norm_eps,
        #    use_high_precision_pow=True,
        #)
        #self.ff_ln = LayerNormParameterized(
        #    self.config.emb_dim,
        #    elementwise_scale=True,
        #    elementwise_shift=False,
        #    use_mean=False,
        #    eps=self.config.layer_norm_eps,
        #    use_high_precision_pow=True,
        #)
        
        #self.attn = MultiHeadAttention(
        #    self.config.emb_dim,
        #    emb_kq,
        #    emb_v,
        #    self.config.nheads,
        #    kvheads,
        #    p_dropout=self.config.hidden_dropout_prob,
        #    use_bias=True,
        #    #position_encoder=rotary_emb,
        #    fused=self.config.fused_weights,
        #    linear_config=self.config.linear_config,
        #)
        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            self.config.emb_dim // self.config.nheads,
            self.config.emb_dim // self.config.nheads,
            self.config.nheads,
            kvheads=1 if self.config.multiquery_attn else self.config.nheads,
            p_dropout=self.config.hidden_dropout_prob,
            use_bias=True,
            fused=self.config.fused_weights,
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
        if self.config.hidden_dropout_prob != 0:
            self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
      
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        residual = x
        x = self.attn(
            q=x,
            **attn_kwargs,
        )
        #print("reached here")
        if self.config.hidden_dropout_prob != 0:
            x = self.dropout(x)
        x = x + residual
        residual = x
        x = self.ln(x)
        x = self.ff_sub_layer(x)
        x = self.ff_ln(x)
        if self.config.hidden_dropout_prob != 0:
            x = self.dropout(x)
        x = x + residual

        return x


class MpnetHeadless(nn.Module):
    def __init__(
        self,
        config: Optional[MpnetConfig] = None,
        **kwargs,
    ):
        super(MpnetHeadless, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = MpnetConfig()
        self.config = self.config.updated(**kwargs)
        self.width = self.config.emb_dim
        self.pad_id = self.config.pad_id
        self.max_expected_seq_len = self.config.max_expected_seq_len

        #self.embedding = nn.Embedding(
        #    self.config.src_vocab_size,
        #    self.config.emb_dim,
        #    padding_idx=self.config.pad_id,
        #)

        self.embedding = nn.Embedding(config.src_vocab_size, config.emb_dim, padding_idx=self.pad_id)
        self.position_embeddings = nn.Embedding(
            config.max_expected_seq_len, config.emb_dim, padding_idx=self.pad_id
        )

        self.enc_norm = nn.LayerNorm(config.emb_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            #"position_ids", torch.arange(config.max_expected_seq_len).expand((1, -1)), persistent=False
            "position_ids", torch.arange(514).expand((1, -1)), persistent=False
        )
        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = MpnetBlock(self.config)
            layers.append(block)
        self.layers = nn.ModuleList(layers)
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, self.config.nheads)
    def compute_position_bias(self, x, position_ids=None, num_buckets=32):
        bsz, qlen, klen = x.size(0), x.size(1), x.size(1)
        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            context_position = torch.arange(qlen, dtype=torch.long)[:, None]
            memory_position = torch.arange(klen, dtype=torch.long)[None, :]

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(relative_position, num_buckets=num_buckets)
        rp_bucket = rp_bucket.to(x.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        return values

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
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)

        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def reset_parameters(self):
        for layer in ["embedding", "position_embedding"]:
            nn.init.normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=self.config.emb_dim**-0.5,
            )
        nn.init.zeros_(self.token_type_embeddings.weight)
        for layer in self.layers:
            for sublayer in ["ln", "ff_ln", "attn", "ff_sub_layer"]:
                getattr(layer, sublayer).reset_parameters()
        self.enc_norm.reset_parameters()

    def forward(
        self,
        x_in,
        #hidden_states: torch.Tensor,
        #attention_mask: Optional[torch.Tensor] = None,
        #head_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        inputs_embeds = self.embedding(x_in)
        if position_ids is None:
            if x_in is not None:
                position_ids = create_position_ids_from_input_ids(x_in, self.pad_id)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if x_in is not None:
            input_shape = x_in.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.enc_norm(embeddings)
        embeddings = self.dropout(embeddings)
        position_bias = self.compute_position_bias(embeddings)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (embeddings,)
        for layer in self.layers:
            x = layer(x=embeddings, position_ids=position_bias,
                #past_key_value_state=past_key_value_states[i],
                #use_cache=use_cache,
                **kwargs)
            if output_attentions:
                all_attentions = all_attentions + (x[1],)

            x = x[0]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
        if not return_dict:
            return tuple(v for v in [x, all_hidden_states, all_attentions] if v is not None)
        #return x




class Mpnet(nn.Module):
     def __init__(
        self,
        config: Optional[MpnetConfig] = None,
        **kwargs,
    ):
        super(Mpnet, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = MpnetConfig()
        self.config = self.config.updated(**kwargs)
        self.base_model = MpnetHeadless(self.config)
        #self.head = nn.Linear(
        #    self.config.emb_dim, self.config.src_vocab_size, bias=False)
        self.dense = nn.Linear(config.emb_dim, config.emb_dim)
        self.activation = nn.Tanh()
     @classmethod
     def from_config(cls, config: MpnetConfig) -> "Mpnet":
         return cls(config)
 
     def get_config(self) -> MpnetConfig:
         return self.config
     def reset_parameters(self):
        self.head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * self.config.src_vocab_size)),
        )
        self.base_model.reset_parameters()

     def post_init(self):
         # This function is called in `get_model` after the model is fully initalized
         # on the correct device
         if self.config.tie_heads:
            # handle assignment of non-meta weights to meta parameters
            if self.head.weight.device == torch.device("meta"):
                self.head.weight = self.base_model.embedding.weight
            else:
                self.base_model.embedding.weight = self.head.weight

         #self.base_model.post_init()


     def forward(
        self,
        x: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: bool = False,
        only_last_token: bool = False,
        head_mask: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **attn_kwargs: Unpack[AttentionKwargs],
        #**kwargs,
    ):
        get_attention_type(**attn_kwargs)["validate_attn_kwargs"](
            input_ids=x, position_ids=position_ids, **attn_kwargs
        )
        # TODO add cache in return
        output = self.base_model(
            x,
            position_ids,
            past_key_value_states,
            use_cache,
            **attn_kwargs,
            #**kwargs,
        )
        return output
        #first_token_tensor = output[0][:, 0]
        #print(first_token_tensor)
        #pooled_output = self.dense(first_token_tensor)
        #pooled_output = self.activation(pooled_output)
        #if not return_dict:
        #    return (first_token_tensor, pooled_output) + output[1:]

        #if only_last_token:
        #    output = output[:, -1, :]
        #preds = self.head(output)
        #preds = preds / self.config.logits_scaling
        #return preds
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`. :param torch.Tensor x: :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx

_architecture_name = "mpnet"


def _mpnet_factory_factory(config):
    def factory(**kwargs):
        return Mpnet(config, **kwargs)

    return factory

_v2_config = MpnetConfig(
    src_vocab_size = 30_527,
    emb_dim = 768,
    nlayers = 12, #12
    nheads = 12, #12
    intermediate_size=3072,
    activation_fn="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_expected_seq_len=512,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    relative_attention_num_buckets=32,
    pad_id=1,
    bos_token_id=0,
    eos_token_id=2,
    fused_weights=False,
)
models.register_model(_architecture_name, "v2", _mpnet_factory_factory(_v2_config))

#def _weight_fusion(
#    input_sd: Mapping, model_config: Optional[MpnetConfig] = None, **kwargs
#):
#    has_fused_weights = True
#    if model_config:
#        if not model_config.fused_weights:
#            has_fused_weights = False
#
#    new_sd = input_sd
#    if has_fused_weights:
#        new_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
#            serialization._attn_unfused_to_fused_step(new_sd)
#        )
#    return new_sd
#
#
#serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)

#(r"attention\.output\.dense", "attn.dense"),

def _hf_to_fms_names(hf_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"embeddings.word_embeddings.weight", "base_model.embedding.weight"),
        (r"^encoder.relative_attention_bias.weight", "base_model.relative_attention_bias.weight"),
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
        (r"output\.LayerNorm", "ln"),
        (r"attention\.LayerNorm", "ff_ln"),
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

        # hf always has the first 2 spots set, we need to remove them as they are not used
        if name == "embeddings.position_embeddings.weight":
            new_sd[new_name] = new_sd[new_name][2:]

    return new_sd

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)
serialization.register_adapter("mpnet", "hf", ["hf_to_fms_names"]) #, "weight_fusion"])
