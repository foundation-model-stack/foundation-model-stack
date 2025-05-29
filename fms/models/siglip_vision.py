import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple, Unpack

import torch
import torch.nn as nn
import numpy as np

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    SDPAAttentionKwargs,
)
from fms.modules.feedforward import FeedForwardBlock
from fms.modules.layernorm import LayerNormParameterized
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig

#TODO: percolate ditributed_strategy, taking care of _no_split_modules from original transformers code

logger = logging.getLogger(__name__)


@dataclass
class SiglipVisionConfig(ModelConfig):
    # Default config yiels vision encoder of the google/siglip-base-patch16-224 model
    hidden_size=768
    intermediate_size=3072
    num_hidden_layers=12
    num_attention_heads=12
    num_channels=3
    image_size=224
    patch_size=16
    hidden_act="gelu-tanh"
    layer_norm_eps=1e-6
    attention_dropout=0.0
    fused_weights=True


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        #self.position_ids = torch.arange(self.num_positions).expand((1, -1))
        # TODO: need to figure out device here, otherwise 'meta'
        self.position_ids = torch.arange(self.num_positions, device='cuda').expand((1, -1))
        #self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)
    
    def reset_parameters(self):
        nn.init.normal_(self.position_embedding.weight, std=1 / np.sqrt(self.config.hidden_size))
        nn.init.zeros_(self.patch_embedding.bias)

        #lecun_normal for conv2d
        tensor = self.patch_embedding.weight
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        variance = 1.0 / fan_in
        nn.init.trunc_normal_(tensor,std=math.sqrt(variance))

    # NOTE: Does not support interpolation of position encodings
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        head_dim = self.embed_dim // self.config.num_attention_heads
        emb_kq = head_dim 
        emb_v = head_dim 
        nheads = self.config.num_attention_heads
        kvheads = self.config.num_attention_heads
        attn_scale_factor = head_dim**-0.5

        self.layer_norm1 = LayerNormParameterized(
            self.embed_dim,
            elementwise_shift=True,
            use_mean=True,
            eps=config.layer_norm_eps,
            use_high_precision_pow=True
        )

        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            emb_kq,
            emb_v,
            nheads,
            kvheads,
            p_dropout=self.config.attention_dropout,
            use_bias=True,
            scale_factor=attn_scale_factor,
        )

        self.layer_norm2 = LayerNormParameterized(
            self.embed_dim,
            elementwise_shift=True,
            use_mean=True,
            eps=config.layer_norm_eps,
            use_high_precision_pow=True
        )

        self.mlp = FeedForwardBlock(
            config.hidden_size,
            hidden_grow_factor=config.intermediate_size // config.hidden_size,
            activation_fn=str_to_activation(config.hidden_act),    #NOTE: using nn.GELU as opposed to nn.functional.gelu() as in HF impl
            use_bias=True,
            p_dropout=self.config.attention_dropout,
        )

    def reset_parameters(self):
        for m in self.modules():
            if (
                isinstance(m, MultiHeadAttention)
                or isinstance(m, LayerNormParameterized)
                or isinstance(m, FeedForwardBlock)
            ):
                m.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        #**kwargs,
        **attn_kwargs: Unpack[SDPAAttentionKwargs],
    ) -> Tuple[torch.FloatTensor]:
        attn_kwargs["attn_name"] = attn_kwargs.get("attn_name", "sdpa_bidirectional")

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(q=hidden_states, **attn_kwargs)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def reset_parameters(self):
        for m in self.layers:
            m.reset_parameters()

    def forward(
        self,
        inputs_embeds,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, **attn_kwargs)
        return hidden_states


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # HF implementation uses PT MHA here, as opposed to SiglipAttnention as in the SiglipEncoderLayer
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)

        self.layernorm = LayerNormParameterized(
            config.hidden_size,
            elementwise_shift=True,
            use_mean=True,
            eps=config.layer_norm_eps,
            use_high_precision_pow=True
        )

        self.mlp = FeedForwardBlock(
            config.hidden_size,
            hidden_grow_factor=config.intermediate_size // config.hidden_size,
            activation_fn=str_to_activation(config.hidden_act),    #NOTE: using nn.GELU as opposed to nn.functional.gelu() as in HF impl
            use_bias=True,
            p_dropout=config.attention_dropout,
        )
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.probe.data)
        nn.init.xavier_uniform_(self.attention.in_proj_weight.data)
        nn.init.zeros_(self.attention.in_proj_bias.data)
        self.layer_norm.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVision(nn.Module):
    def __init__(
        self,
        config: SiglipVisionConfig = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(SiglipVision, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = SiglipVisionConfig()
        self.config = self.config.updated(**kwargs)

        self.distributed_strategy = distributed_strategy
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = LayerNormParameterized(
            config.hidden_size,
            elementwise_shift=True,
            use_mean=True,
            eps=config.layer_norm_eps,
            use_high_precision_pow=True,
        )
        self.use_head = True if not hasattr(config, "vision_use_head") else config.vision_use_head
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config)

    @classmethod
    def from_config(cls, config: SiglipVisionConfig) -> "SiglipVision":
        return cls(config)

    def get_config(self) -> SiglipVisionConfig:
        return self.config

    def reset_parameters(self):
        for m in self.modules():
            m.reset_parameters()

    def forward(
        self,
        pixel_values,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(inputs_embeds=hidden_states, **attn_kwargs)
        hidden_states = self.post_layernorm(hidden_states)
        pooler_output = self.head(hidden_states) if self.use_head else None
        return hidden_states, pooler_output


_siglip_base_patch16_224_config = SiglipVisionConfig()

_architecture_name = "siglip_vision"

def _siglip_vision_factory_factory(config):
    def factory(**kwargs):
        return SiglipVision(config, **kwargs)
    return factory

models.register_model(
    _architecture_name, "siglip_base_patch16_224", _siglip_vision_factory_factory(_siglip_base_patch16_224_config)
)


def _weight_fusion(
    input_sd: Mapping, model_config: Optional[SiglipVisionConfig] = None, **kwargs
):
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        new_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
            serialization._attn_unfused_to_fused_step(new_sd)
        )
    return new_sd

def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"vision_model\.head","head"),
        (r"^vision_model\.encoder","encoder"),
        (r"vision_model\.embeddings","embeddings"),
        (r"vision_model\.post_layernorm","post_layernorm"),
        (r"self_attn\.k_proj", "self_attn.in_proj.key"),
        (r"self_attn\.v_proj", "self_attn.in_proj.value"),
        (r"self_attn\.q_proj", "self_attn.in_proj.query"),
        (r"self_attn\.out_proj", "self_attn.dense"),
        (r"mlp\.fc1", "mlp.w1"),
        (r"mlp\.fc2", "mlp.w2"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd    

serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "weight_fusion"],
)
