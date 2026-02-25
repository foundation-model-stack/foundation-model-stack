import math
import logging
from dataclasses import dataclass

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils.config import ModelConfig
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
)
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import PixtralRotaryEmbedding
from fms.utils.activation import str_to_activation

import torch
from torch import nn
from typing import Any, Unpack


logger = logging.getLogger(__name__)


def get_positions_in_meshgrid(
    patch_embeds_list: list[torch.Tensor],
) -> torch.Tensor:
    """Get the 2D coordinates for each patch.

    NOTE: Transformers collapses the position IDs to 1D and flattens
    freqs; our implementation for Pixtral Rope is based on Mistral inference
    since it aligns better with other rope implementations in FMS.
    """
    positions = []
    for patch in patch_embeds_list:
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        # IDs are 2D for pixtral Rope
        pos_id = torch.stack(mesh, dim=-1).reshape(-1, 2)
        positions.append(pos_id)

    # Add batch dimension to match patch_embeds shape
    return torch.cat(positions).unsqueeze(0)


@dataclass
class PixtralVisionConfig(ModelConfig):
    # Identical configuration to the vision encoder in
    # mistralai/Mistral-Small-3.2-24B-Instruct-2506
    hidden_size: int = 1024
    intermediate_size: int = 4096
    nlayers: int = 24
    nheads: int = 16
    nchannels: int = 3
    image_size: int = 1540
    patch_size: int = 14
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    # FMS specific
    linear_config: dict[str, Any] | None = None
    fused_weights: bool = True


class PixtralRMSNorm(LayerNormParameterized):
    """Pixtral's RMS Norm using the FMS implementation of LayerNorm.
    Note that LayerNormParameterized implements parameter reset.
    """

    def __init__(self, normalized_shape: int, eps: float):
        super().__init__(
            normalized_shape,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=eps,
            use_high_precision_pow=True,
        )


class PixtralAttentionLayer(nn.Module):
    def __init__(self, config: PixtralVisionConfig, rotary_emb: PixtralRotaryEmbedding):
        super().__init__()
        self.config = config
        head_dim = self.config.hidden_size // self.config.nheads
        mlp_grow_factor = self.config.intermediate_size / self.config.hidden_size

        # Attention related
        self.attention_norm = PixtralRMSNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.layer_norm_eps,
        )
        self.attn = MultiHeadAttention(
            emb_dim=self.config.hidden_size,
            emb_kq=head_dim,
            emb_v=head_dim,
            nheads=config.nheads,
            kvheads=config.nheads,
            p_dropout=self.config.attention_dropout,
            position_encoder=rotary_emb,
            linear_config=self.config.linear_config,
            fused=self.config.fused_weights,
            scale_factor=head_dim**-0.5,
        )

        # Feedforward related
        self.ffn_norm = PixtralRMSNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.layer_norm_eps,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.hidden_size,
            hidden_grow_factor=mlp_grow_factor,
            activation_fn=str_to_activation(self.config.hidden_act),
            use_bias=False,
            p_dropout=0,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attn(
            q=hidden_states,
            position_ids=position_ids,
            **attn_kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ff_sub_layer(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def reset_parameters(self):
        for m in self.modules():
            if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
                m.reset_parameters()


class PixtralTransformer(nn.Module):
    def __init__(self, config: PixtralVisionConfig, rotary_emb: PixtralRotaryEmbedding):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [PixtralAttentionLayer(config, rotary_emb) for _ in range(config.nlayers)]
        )

    def forward(
        self,
        inputs_embeds,
        position_ids: torch.Tensor,
        output_hidden_states=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        hidden_states = inputs_embeds
        # TODO: Currently aligns with siglip to return an empty tuple
        # when there are no hidden states, but None would make more sense
        # here and better align with HF Transformers for readability.
        encoder_states = (hidden_states,) if output_hidden_states else ()

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                **attn_kwargs,
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        return hidden_states, encoder_states

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters") and callable(layer.reset_parameters):
                layer.reset_parameters()


class PixtralVisionModel(nn.Module):
    def __init__(
        self,
        config: PixtralVisionConfig | None = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = PixtralVisionConfig()
        self.config = self.config.updated(**kwargs)
        self.patch_size = self.config.patch_size

        self.distributed_strategy = distributed_strategy
        self.patch_conv = nn.Conv2d(
            in_channels=self.config.nchannels,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            bias=False,
        )
        self.ln_pre = PixtralRMSNorm(
            normalized_shape=self.config.hidden_size,
            eps=self.config.layer_norm_eps,
        )

        head_dim = self.config.hidden_size // self.config.nheads
        self.patch_positional_embedding = PixtralRotaryEmbedding(
            dim=head_dim,
            theta=self.config.rope_theta,
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
        )

        self.transformer = PixtralTransformer(
            self.config,
            self.patch_positional_embedding,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor | list[tuple[int, int]],
        output_hidden_states=False,
        position_ids=None,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        pixel_values = pixel_values.to(dtype=self.patch_conv.weight.dtype)

        # Pass images through initial convolution independently + flatten
        patch_embeds = self.patch_conv(pixel_values)

        # Force divisibility by patch size
        patch_embeds_list = [
            embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
            for embed, size in zip(patch_embeds, image_sizes)
        ]

        patch_embeds = torch.cat(
            [p.flatten(1).T for p in patch_embeds_list], dim=0
        ).unsqueeze(0)

        patch_embeds = self.ln_pre(patch_embeds)

        # Similar to the Mistral Commons implementation of Pixtral,
        # we keep the position IDs 2D since it's a bit easier to read
        # for the interleaved implementation of RoPE.
        position_ids = get_positions_in_meshgrid(
            patch_embeds_list,
        )

        # Invoke the actual transformer
        return self.transformer(
            patch_embeds,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            **attn_kwargs,
        )

    def _clean_up_rot_emb_cache(
        self,
        cached_freqs: dict[int, torch.Tensor],
    ):
        # Ensure that are no cached freqs on meta device for any reason
        for dev in list(cached_freqs.keys()):
            if cached_freqs[dev].device == torch.device("meta"):
                del cached_freqs[dev]

    @classmethod
    def from_config(cls, config: PixtralVisionConfig) -> "PixtralVisionModel":
        return cls(config)

    def get_config(self) -> PixtralVisionConfig:
        return self.config

    def post_init(self):
        # This function is called in `get_model` after the model is
        # fully initalized on the correct device
        self._clean_up_rot_emb_cache(
            self.patch_positional_embedding.cached_freqs,
        )

        # init RoPE on the right device(s)
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.patch_positional_embedding.compute_freqs_cis(device)

    def reset_parameters(self):
        # Conv2D - use lecun_normal with no bias, copied from FMS siglip
        tensor = self.patch_conv.weight
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        variance = 1.0 / fan_in
        nn.init.trunc_normal_(tensor, std=math.sqrt(variance))

        if self.patch_conv.bias:
            nn.init.zeros_(self.patch_conv.bias)

        self.ln_pre.reset_parameters()
        self.transformer.reset_parameters()

        # Reinitialize the 2D RoPE
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.patch_positional_embedding.compute_freqs_cis(device)


# NOTE: We do not currently offer support for Pixtral as a standalone
# vision encoder, as this model is largely used in the composite
# architecture for Mistral3 within FMS. While there are standalone vision
# models for pixtral, they primarily use Mistral's model format instead
# of HF Transformers, and would need to be converted for direct use in FMS.
#
# If the need for pixtral to run as a standalone vision encoder is pressing
# in the future, we can take the normal pattern and add a factory for it,
# then port the vision parts of the adapters from Mistral3 here.
