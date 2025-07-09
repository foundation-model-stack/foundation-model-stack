import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Unpack

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
)
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.modules.linear import get_linear_type
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class PixtralVisionConfig(ModelConfig):
    hidden_size: int = 1024
    intermediate_size: int = 4096
    nlayers: int = 24
    nheads: int = 16
    nchannels: int = 3
    image_size: int = 1024
    patch_size: int = 16
    hidden_act: str = "silu"
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    linear_config: Optional[Mapping[str, Any]] = None
    fused_weights: bool = True


class PixtralAttentionLayer(nn.Module):
    def __init__(self, config: PixtralVisionConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        head_dim = config.hidden_size // config.nheads
        emb_kq = head_dim
        emb_v = head_dim
        nheads = config.nheads
        kvheads = config.nheads
        attn_scale_factor = head_dim**-0.5

        self.attn = MultiHeadAttention(
            config.hidden_size,
            emb_kq,
            emb_v,
            nheads,
            kvheads,
            p_dropout=config.attention_dropout,
            use_bias=False,
            position_encoder=rotary_emb,
            fused=config.fused_weights,
            linear_config=config.linear_config,
            scale_factor=attn_scale_factor,
        )

        self.attention_norm = LayerNormParameterized(
            config.hidden_size,
            elementwise_shift=False,
            use_mean=False,
            eps=config.layer_norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_sub_layer = GatedLinearUnit(
            config.hidden_size,
            hidden_grow_factor=config.intermediate_size / config.hidden_size,
            activation_fn=str_to_activation(config.hidden_act),
            use_bias=False,
            p_dropout=config.attention_dropout,
            fused=config.fused_weights,
            linear_config=config.linear_config,
        )
        self.ffn_norm = LayerNormParameterized(
            config.hidden_size,
            elementwise_shift=False,
            use_mean=False,
            eps=config.layer_norm_eps,
            use_high_precision_pow=True,
        )

    def reset_parameters(self):
        for m in self.modules():
            if (
                isinstance(m, MultiHeadAttention)
                or isinstance(m, LayerNormParameterized)
                or isinstance(m, GatedLinearUnit)
            ):
                m.reset_parameters()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        attn_kwargs["attn_name"] = attn_kwargs.get("attn_name", "sdpa_bidirectional")

        residual = hidden_states

        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attn(
            q=hidden_states,
            position_ids=position_ids,
            mask=attention_mask,
            **attn_kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ff_sub_layer(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class PixtralTransformer(nn.Module):
    def __init__(self, config: PixtralVisionConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList()
        for _ in range(config.nlayers):
            self.layers.append(PixtralAttentionLayer(config, rotary_emb))

    def reset_parameters(self):
        for m in self.layers:
            m.reset_parameters()

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        hidden_states = inputs_embeds
        encoder_states = (hidden_states,) if output_hidden_states else ()

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                position_ids=position_ids,
                **attn_kwargs,
            )
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

        return hidden_states, encoder_states


class PixtralVision(nn.Module):
    def __init__(
        self,
        config: Optional[PixtralVisionConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(PixtralVision, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = PixtralVisionConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.patch_conv = nn.Conv2d(
            in_channels=self.config.nchannels,
            out_channels=self.config.hidden_size,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
            bias=False,
        )
        self.patch_size = self.config.patch_size
        self.image_size = self.config.image_size

        rope_scaling = {
            "rope_type": "pixtral",
            "max_patches_per_side": self.image_size // self.patch_size,
        }
        self.rot_emb = RotaryEmbedding(
            dim=self.config.hidden_size // self.config.nheads,
            scaling=rope_scaling,
            ratio=self.config.rope_theta,
        )
        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device)

        self.transformer = PixtralTransformer(self.config, self.rot_emb)

        self.ln_pre = LayerNormParameterized(
            self.config.hidden_size,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.layer_norm_eps,
            use_high_precision_pow=True,
        )

    @classmethod
    def from_config(cls, config: PixtralVisionConfig) -> "PixtralVision":
        return cls(config)

    def get_config(self) -> PixtralVisionConfig:
        return self.config

    def reset_parameters(self):
        for m in self.modules():
            m.reset_parameters()

    def _clean_up_rot_emb_cache(
        self,
        cached_freqs: dict[Optional[torch.device], dict[int, torch.Tensor]],
        max_seq_len_cached: dict[Optional[torch.device], int],
    ):
        # remove meta tensors from cached_freqs
        for dev in list(cached_freqs.keys()):
            for alp in list(cached_freqs[dev].keys()):
                if cached_freqs[dev][alp].device == torch.device("meta"):
                    del cached_freqs[dev][alp]
                    if len(cached_freqs[dev]) == 0:
                        del cached_freqs[dev]
                        del max_seq_len_cached[dev]

    def post_init(self):
        # This function is called in `get_model` after the model is
        # fully initalized on the correct device

        self._clean_up_rot_emb_cache(
            self.rot_emb.cached_freqs,
            self.rot_emb.max_seq_len_cached,
        )

        # init RoPE on the right device(s)
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device)

    def generate_block_attention_mask(self, patch_embeds_list, tensor):
        dtype = tensor.dtype
        device = tensor.device
        seq_len = tensor.shape[1]
        d_min = torch.finfo(dtype).min
        causal_mask = torch.full(
            (seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device
        )

        block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
        block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
        for start, end in zip(block_start_idx, block_end_idx):
            causal_mask[start:end, start:end] = 0

        causal_mask = causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)
        return causal_mask

    def position_ids_in_meshgrid(self, patch_embeds_list, max_width):
        positions = []
        for patch in patch_embeds_list:
            height, width = patch.shape[-2:]
            mesh = torch.meshgrid(
                torch.arange(height), torch.arange(width), indexing="ij"
            )
            h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
            ids = h_grid * max_width + v_grid
            positions.append(ids[:, 0])
        return torch.cat(positions)

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        *args,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        # pass images through initial convolution independently
        patch_embeds = self.patch_conv(pixel_values)
        # Potential graph break here:
        patch_embeds_list = [
            embed[..., : (size[0] // self.patch_size), : (size[1] // self.patch_size)]
            for embed, size in zip(patch_embeds, image_sizes)
        ]

        # flatten to a single sequence
        patch_embeds = torch.cat(
            [p.flatten(1).T for p in patch_embeds_list], dim=0
        ).unsqueeze(0)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        position_ids = self.position_ids_in_meshgrid(
            patch_embeds_list,
            max_width=self.config.image_size // self.config.patch_size,
        )
        position_ids = position_ids.unsqueeze(0)

        attention_mask = self.generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], patch_embeds
        )

        last_hidden_state, hidden_states = self.transformer(
            patch_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            **attn_kwargs,
        )

        if output_hidden_states:
            return last_hidden_state, hidden_states

        return last_hidden_state


_pixtral_12b_vision_config = PixtralVisionConfig()

_architecture_name = "pixtral_vision"


def __pixtral_factory_factory(config):
    def factory(**kwargs):
        return PixtralVision(config, **kwargs)

    return factory


models.register_model(
    _architecture_name,
    "_pixtral_12b_vision",
    __pixtral_factory_factory(_pixtral_12b_vision_config),
)


def _weight_fusion(
    input_sd: Mapping, model_config: Optional[PixtralVisionConfig] = None, **kwargs
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
        (r"attention\.k_proj", "attn.in_proj.key"),
        (r"attention\.v_proj", "attn.in_proj.value"),
        (r"attention\.q_proj", "attn.in_proj.query"),
        (r"attention\.o_proj", "attn.dense"),
        (r"feed_forward\.gate_proj", "ff_sub_layer.wg"),
        (r"feed_forward\.up_proj", "ff_sub_layer.w1"),
        (r"feed_forward\.down_proj", "ff_sub_layer.w2"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    if "int8" in linear_type:
        # quantize_weight is fms-model-optimizer identifier of weight clip values
        return ["weight", "bias", "quantize_weight"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any],
    model_config=None,
    **kwargs,
) -> Mapping[str, Any]:
    new_sd = {}

    if model_config:
        head_size = model_config.emb_dim // model_config.nheads
        linear_type_str = "torch_linear"
        if model_config.linear_config:
            linear_type_str = get_linear_type(
                model_config.linear_config,
                module_name=None,  # if callable, linear_type should return default str
            )
    else:
        logger.warning("Missing model_config, assuming defaults for head_size")
        head_size = 128  # Good default for most models
        linear_type_str = "torch_linear"

    rope_params = _get_rope_params(linear_type_str)
    trans_required_pattern = re.compile(
        f"layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})"
    )
    for name, param in input_sd.items():
        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters for Llama models
        # This transpose is due to the different implementation of RoPE in
        # HF and FMS. While FMS follows the original RoPE paper
        # (https://arxiv.org/abs/2104.09864), HF has its own implementation
        # that doesn't respect the order of outputs. This is OK as long as you
        # rearrange the weights of the query and key projections, as the
        # combination projection + RoPE ends up producing the same outputs.
        # Therefore, to make FMS produce the correct order of outputs when
        # loading from an HF checkpoint, we need to undo the transformation
        # that HF does from the original Meta weights
        is_gptq_2d_qparam = "gptq" in linear_type_str and param.dim() == 2
        if bool(trans_required_pattern.match(name)) and param.numel() > 1:
            temp = param
            if is_gptq_2d_qparam:
                # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
                # and are fully transposed before & after process.
                # GPTQ scales and qzeros are also transposed accordingly
                temp = temp.transpose(0, 1)
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # 1-dim parameters
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            if is_gptq_2d_qparam:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "weight_fusion"],
)
