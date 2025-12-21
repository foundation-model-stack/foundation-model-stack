import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


@dataclass
class ConformerConfig(ModelConfig):
    """Configuration for Conformer encoder used in speech processing."""
    # Defaults match HF GraniteSpeechEncoderConfig
    num_features: int = 160  # 80 log-mel * 2 channels
    hidden_dim: int = 1024
    num_layers: int = 16
    num_heads: int = 8
    dim_head: int = 128
    conv_kernel_size: int = 15
    conv_expansion_factor: int = 2
    feedforward_mult: int = 4
    dropout: float = 0.1
    max_pos_emb: int = 512
    context_size: int = 200  # Blocked attention context window
    output_dim: int = 256  # CTC vocabulary size
    use_ctc: bool = True
    activation: str = "silu"
    linear_config: Optional[Mapping[str, Any]] = None


class ConformerFeedForward(nn.Module):
    """Macaron-style feedforward with pre-normalization (used before and after attention)."""

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.mult = mult
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mult)
        self.activation = str_to_activation(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class ConformerAttention(nn.Module):
    """Multi-head self-attention with Shaw relative positional encoding and blocked computation."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        max_pos_emb: int = 512,
        context_size: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.max_pos_emb = max_pos_emb
        self.context_size = context_size
        self.dropout_prob = dropout
        self.scale = dim_head ** -0.5

        if context_size <= 0 or context_size > max_pos_emb:
            raise ValueError("Context size is either less than 0 or exceeds the max_pos_emb")

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)  # Shaw relative positions
        self.to_out = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_dists: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm(x)
        bsz, num_features, _ = x.shape

        # Pad to block size
        num_blocks = math.ceil(num_features / self.context_size)
        remainder = num_features % self.context_size
        if remainder > 0:
            x = F.pad(x, (0, 0, 0, self.context_size - remainder))

        # Project to Q, K, V
        query_states = self.to_q(x)
        key_states, value_states = self.to_kv(x).chunk(2, dim=-1)

        # Reshape for blocked multi-head attention
        query_states = query_states.reshape(bsz, num_blocks, self.context_size, self.num_heads, -1).transpose(2, 3)
        key_states = key_states.reshape(bsz, num_blocks, self.context_size, self.num_heads, -1).transpose(2, 3)
        value_states = value_states.reshape(bsz, num_blocks, self.context_size, self.num_heads, -1).transpose(2, 3)

        # Shaw's relative positional embedding
        rel_pos_emb = self.pos_emb(attention_dists)
        pos_attn = torch.einsum("b m h c d, c r d -> b m h c r", query_states, rel_pos_emb) * self.scale

        # Mask padded positions in last block
        if remainder > 0:
            mask = torch.ones(self.context_size, self.context_size, dtype=torch.bool, device=x.device)
            mask[:remainder, :remainder] = False
            mask_value = -torch.finfo(pos_attn.dtype).max
            pos_attn[:, -1, :].masked_fill_(mask, mask_value)

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = F.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=pos_attn, scale=self.scale
            )

        out = out.transpose(2, 3).reshape(bsz, x.shape[1], -1)
        out = self.to_out(out[:, :num_features, :])
        return self.dropout(out)


class ConformerConvModule(nn.Module):
    """Depthwise-separable convolution module with GLU activation."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * expansion_factor * 2, kernel_size=1)  # 2x for GLU
        self.depthwise_conv = nn.Conv1d(
            dim * expansion_factor,
            dim * expansion_factor,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=dim * expansion_factor,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(dim * expansion_factor)
        self.activation = str_to_activation(activation)
        self.pointwise_conv2 = nn.Conv1d(dim * expansion_factor, dim, kernel_size=1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # GLU gating
        x, gate = x.chunk(2, dim=1)
        x = x * torch.sigmoid(gate)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        x = self.dropout_layer(x)
        return x


class ConformerBlock(nn.Module):
    """Conformer block with Macaron-style half-step feedforward residuals."""

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config

        # Macaron-style: FF before and after attention
        self.ff1 = ConformerFeedForward(
            dim=config.hidden_dim,
            mult=config.feedforward_mult,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.attn = ConformerAttention(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            dim_head=config.dim_head,
            max_pos_emb=config.max_pos_emb,
            context_size=config.context_size,
            dropout=config.dropout,
        )
        self.conv = ConformerConvModule(
            dim=config.hidden_dim,
            kernel_size=config.conv_kernel_size,
            expansion_factor=config.conv_expansion_factor,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.ff2 = ConformerFeedForward(
            dim=config.hidden_dim,
            mult=config.feedforward_mult,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.post_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_dists: torch.Tensor,
    ) -> torch.Tensor:
        # Half-step residual for FF, full residual for attn/conv
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn(x, attention_dists)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.post_norm(x)
        return x


class ConformerEncoder(nn.Module):
    """Conformer encoder for acoustic feature extraction from mel-spectrograms."""

    def __init__(
        self,
        config: Optional[ConformerConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = ConformerConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.input_proj = nn.Linear(self.config.num_features, self.config.hidden_dim)
        self.blocks = nn.ModuleList([
            ConformerBlock(self.config) for _ in range(self.config.num_layers)
        ])

        # Optional CTC (Connectionist Temporal Classification) auxiliary loss
        if self.config.use_ctc:
            self.out = nn.Linear(self.config.hidden_dim, self.config.output_dim)
            self.out_mid = nn.Linear(self.config.output_dim, self.config.hidden_dim)
        else:
            self.out = None
            self.out_mid = None

        # Non-persistent buffer for relative position distances
        attention_dists = self._precompute_attention_dists()
        self.register_buffer("attention_dists", attention_dists, persistent=False)

    def _recompute_buffers(self):
        # Recompute non-persistent buffers after meta device transfer
        device = next(self.parameters()).device
        if device.type != "meta":
            attention_dists = self._precompute_attention_dists(device)
            self.register_buffer("attention_dists", attention_dists, persistent=False)

    @classmethod
    def from_config(cls, config: ConformerConfig) -> "ConformerEncoder":
        return cls(config)

    def get_config(self) -> ConformerConfig:
        return self.config

    def _precompute_attention_dists(self, device: torch.device = None) -> torch.Tensor:
        context_size = self.config.context_size
        max_pos_emb = self.config.max_pos_emb
        seq = torch.arange(context_size, device=device)
        relpos_dist = seq.view(-1, 1) - seq.view(1, -1)
        attention_dists = torch.clamp(relpos_dist, -context_size, context_size) + max_pos_emb
        return attention_dists.long()

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features = input_features.shape
        assert num_features == self.config.num_features, (
            f"Input features dimension {num_features} doesn't match "
            f"config.num_features {self.config.num_features}"
        )

        x = self.input_proj(input_features)
        attention_dists = self.attention_dists

        # Intermediate CTC feedback: inject softmax predictions at mid-layer for auxiliary loss
        mid_layer = len(self.blocks) // 2
        for idx, block in enumerate(self.blocks, start=1):
            x = block(x, attention_dists)
            if self.config.use_ctc and self.out is not None and idx == mid_layer:
                x_mid = x.clone()
                x_mid = self.out(x_mid)
                x = x + self.out_mid(F.softmax(x_mid, dim=-1))

        return x


_architecture_name = "conformer"
_default_config = ConformerConfig()


def _conformer_factory_factory(config):
    def factory(**kwargs):
        return ConformerEncoder(config, **kwargs)
    return factory


models.register_model(
    _architecture_name,
    "granite_speech",
    _conformer_factory_factory(_default_config),
)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"^encoder\.input_linear\.", "input_proj."),
        (r"^encoder\.layers\.(\d+)\.", r"blocks.\1."),
        (r"\.ff1\.pre_norm\.", ".ff1.norm."),
        (r"\.ff1\.up_proj\.", ".ff1.fc1."),
        (r"\.ff1\.down_proj\.", ".ff1.fc2."),
        (r"\.ff2\.pre_norm\.", ".ff2.norm."),
        (r"\.ff2\.up_proj\.", ".ff2.fc1."),
        (r"\.ff2\.down_proj\.", ".ff2.fc2."),
        (r"\.attn\.pre_norm\.", ".attn.norm."),
        (r"\.attn\.to_q\.", ".attn.to_q."),
        (r"\.attn\.to_kv\.", ".attn.to_kv."),
        (r"\.attn\.to_out\.", ".attn.to_out."),
        (r"\.attn\.rel_pos_emb\.", ".attn.pos_emb."),
        (r"\.conv\.pre_norm\.", ".conv.norm."),
        (r"\.conv\.up_conv\.", ".conv.pointwise_conv1."),
        (r"\.conv\.depth_conv\.conv\.", ".conv.depthwise_conv."),
        (r"\.conv\.down_conv\.", ".conv.pointwise_conv2."),
        (r"\.conv\.batch_norm\.", ".conv.batch_norm."),
        (r"\.post_norm\.", ".post_norm."),
        (r"^encoder\.out\.", "out."),
        (r"^encoder\.out_mid\.", "out_mid."),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


def _weight_fusion(
    input_sd: Mapping[str, Any],
    model_config: Optional[ConformerConfig] = None,
    **kwargs
) -> Mapping[str, Any]:
    # Conformer doesn't use fused weights
    return input_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)
serialization.register_adapter_step(
    _architecture_name, "weight_fusion", _weight_fusion
)
serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "weight_fusion"],
)