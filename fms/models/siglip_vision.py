import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Unpack

import torch
import torch.nn as nn
import numpy as np

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
)
from fms.modules.feedforward import FeedForwardBlock
from fms.modules.layernorm import LayerNormParameterized
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig

# TODO: percolate ditributed_strategy, taking care of _no_split_modules from original transformers code

logger = logging.getLogger(__name__)


@dataclass
class SiglipVisionConfig(ModelConfig):
    # Default config yields vision encoder of the google/siglip-base-patch16-224 model
    hidden_size: int = 768
    intermediate_size: int = 3072
    nlayers: int = 12
    nheads: int = 12
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    hidden_act: str = "gelu-tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    linear_config: Optional[Mapping[str, Any]] = None
    fused_weights: bool = True


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
        self.position_ids = torch.arange(self.num_positions).expand((1, -1))

    def reset_parameters(self):
        nn.init.normal_(
            self.position_embedding.weight, std=1 / np.sqrt(self.config.hidden_size)
        )
        nn.init.zeros_(self.patch_embedding.bias)

        # lecun_normal for conv2d
        tensor = self.patch_embedding.weight
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        variance = 1.0 / fan_in
        nn.init.trunc_normal_(tensor, std=math.sqrt(variance))

    def post_init(self):
        device = self.position_embedding.weight.device
        self.position_ids = torch.arange(self.num_positions, device=device).expand(
            (1, -1)
        )

    # NOTE: Does not support interpolation of position encodings-- not used by granite-vision
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        head_dim = self.embed_dim // self.config.nheads
        emb_kq = head_dim
        emb_v = head_dim
        nheads = self.config.nheads
        kvheads = self.config.nheads
        attn_scale_factor = head_dim**-0.5

        self.layer_norm1 = LayerNormParameterized(
            self.embed_dim,
            elementwise_shift=True,
            use_mean=True,
            eps=config.layer_norm_eps,
            use_high_precision_pow=True,
        )

        self.attn = MultiHeadAttention(
            self.embed_dim,
            emb_kq,
            emb_v,
            nheads,
            kvheads,
            p_dropout=self.config.attention_dropout,
            use_bias=True,
            linear_config=self.config.linear_config,
            fused=self.config.fused_weights,
            scale_factor=attn_scale_factor,
        )

        self.layer_norm2 = LayerNormParameterized(
            self.embed_dim,
            elementwise_shift=True,
            use_mean=True,
            eps=config.layer_norm_eps,
            use_high_precision_pow=True,
        )

        self.mlp = FeedForwardBlock(
            config.hidden_size,
            hidden_grow_factor=config.intermediate_size / config.hidden_size,
            activation_fn=str_to_activation(
                config.hidden_act
            ),  # NOTE: using nn.GELU as opposed to nn.functional.gelu() as in HF impl
            use_bias=True,
            linear_config=self.config.linear_config,
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
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        attn_kwargs["attn_name"] = attn_kwargs.get("attn_name", "sdpa_bidirectional")

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attn(q=hidden_states, **attn_kwargs)
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
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.nlayers)]
        )

    def reset_parameters(self):
        for m in self.layers:
            m.reset_parameters()

    def forward(
        self,
        inputs_embeds,
        output_hidden_states=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        hidden_states = inputs_embeds
        encoder_states = (hidden_states,) if output_hidden_states else ()

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, **attn_kwargs)
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        return hidden_states, encoder_states


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # HF implementation uses PT MHA here, as opposed to SiglipAttnention as in the SiglipEncoderLayer
        self.attention = torch.nn.MultiheadAttention(
            config.hidden_size, config.nheads, batch_first=True
        )

        self.layernorm = LayerNormParameterized(
            config.hidden_size,
            elementwise_shift=True,
            use_mean=True,
            eps=config.layer_norm_eps,
            use_high_precision_pow=True,
        )

        self.mlp = FeedForwardBlock(
            config.hidden_size,
            hidden_grow_factor=config.intermediate_size / config.hidden_size,
            activation_fn=str_to_activation(
                config.hidden_act
            ),  # NOTE: using nn.GELU as opposed to nn.functional.gelu() as in HF impl
            use_bias=True,
            p_dropout=config.attention_dropout,
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.probe.data)
        nn.init.xavier_uniform_(self.attention.in_proj_weight.data)
        nn.init.zeros_(self.attention.in_proj_bias.data)
        self.layernorm.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionHeadless(nn.Module):
    def __init__(
        self,
        config: Optional[SiglipVisionConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(SiglipVisionHeadless, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = SiglipVisionConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.embeddings = SiglipVisionEmbeddings(self.config)
        self.encoder = SiglipEncoder(self.config)
        self.post_layernorm = LayerNormParameterized(
            self.config.hidden_size,
            elementwise_shift=True,
            use_mean=True,
            eps=self.config.layer_norm_eps,
            use_high_precision_pow=True,
        )

    def reset_parameters(self):
        for m in self.modules():
            m.reset_parameters()

    def post_init(self):
        self.embeddings.post_init()

    def forward(
        self,
        pixel_values,
        output_hidden_states=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state, hidden_states = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            **attn_kwargs,
        )
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state, hidden_states


class SiglipVision(nn.Module):
    def __init__(
        self,
        config: Optional[SiglipVisionConfig] = None,
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

        self.base_model = SiglipVisionHeadless(self.config, self.distributed_strategy)

        self.head = SiglipMultiheadAttentionPoolingHead(self.config)

    @classmethod
    def from_config(cls, config: SiglipVisionConfig) -> "SiglipVision":
        return cls(config)

    def get_config(self) -> SiglipVisionConfig:
        return self.config

    def reset_parameters(self):
        self.head.reset_parameters()
        self.base_model.reset_parameters()

    def post_init(self):
        self.base_model.post_init()

    def forward(
        self,
        pixel_values,
        output_hidden_states=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        last_hidden_state, hidden_states = self.base_model(
            pixel_values, output_hidden_states=output_hidden_states, **attn_kwargs
        )
        pooler_output = self.head(last_hidden_state)
        if output_hidden_states:
            return last_hidden_state, pooler_output, hidden_states
        return last_hidden_state, pooler_output


_siglip_base_patch16_224_config = SiglipVisionConfig()

_architecture_name = "siglip_vision"


def _siglip_vision_factory_factory(config):
    def factory(**kwargs):
        return SiglipVision(config, **kwargs)

    return factory


models.register_model(
    _architecture_name,
    "siglip_base_patch16_224",
    _siglip_vision_factory_factory(_siglip_base_patch16_224_config),
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
        new_sd = serialization._attn_unfused_to_fused_step(new_sd)
    return new_sd


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"vision_model\.head", "head"),
        (r"^vision_model\.encoder", "base_model.encoder"),
        (r"vision_model\.embeddings", "base_model.embeddings"),
        (r"vision_model\.post_layernorm", "base_model.post_layernorm"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.out_proj", "attn.dense"),
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
