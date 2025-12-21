import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


@dataclass
class SpeechProjectorConfig(ModelConfig):
    """Configuration for Q-Former projector that bridges encoder and decoder dimensions."""
    encoder_dim: int = 1024  # Conformer encoder output dimension
    encoder_hidden_size: Optional[int] = None
    decoder_dim: int = 2048  # Target language model dimension
    window_size: int = 15
    downsample_rate: int = 5
    num_queries: int = 3
    num_hidden_layers: int = 2
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    cross_attention_frequency: int = 1  # Apply cross-attention every N layers
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02


class QFormerSelfAttention(nn.Module):
    """Self-attention layer for query tokens to communicate with each other."""

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        assert config.encoder_dim % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.encoder_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.encoder_dim, self.all_head_size)
        self.key = nn.Linear(config.encoder_dim, self.all_head_size)
        self.value = nn.Linear(config.encoder_dim, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / (self.attention_head_size ** 0.5)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context_layer = torch.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_shape)
        return context_layer


class QFormerCrossAttention(nn.Module):
    """Cross-attention layer where queries attend to encoder outputs."""

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        assert config.encoder_dim % config.num_attention_heads == 0

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.encoder_dim // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        encoder_hidden_size = config.encoder_hidden_size or config.encoder_dim

        self.query = nn.Linear(config.encoder_dim, self.all_head_size)
        self.key = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(encoder_hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query_layer = self._transpose_for_scores(self.query(query_states))
        key_layer = self._transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self._transpose_for_scores(self.value(encoder_hidden_states))

        attn_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_scores = attn_scores / (self.attention_head_size ** 0.5)

        if encoder_attention_mask is not None:
            attn_scores = attn_scores + encoder_attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context_layer = torch.matmul(attn_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer


class QFormerAttentionOutput(nn.Module):
    """Attention output projection with residual connection and layer norm."""

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        self.dense = nn.Linear(config.encoder_dim, config.encoder_dim)
        self.LayerNorm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QFormerIntermediate(nn.Module):
    """Feedforward intermediate layer with activation."""

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        self.dense = nn.Linear(config.encoder_dim, config.intermediate_size)
        self.intermediate_act_fn = str_to_activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class QFormerOutput(nn.Module):
    """Feedforward output layer with residual connection and layer norm."""

    def __init__(self, config: SpeechProjectorConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.encoder_dim)
        self.LayerNorm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QFormerLayer(nn.Module):
    """Q-Former layer with self-attention, optional cross-attention, and feedforward."""

    def __init__(self, config: SpeechProjectorConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        self.self_attention = QFormerSelfAttention(config)
        self.self_attention_output = QFormerAttentionOutput(config)

        # Apply cross-attention every N layers (default: every layer)
        if layer_idx % config.cross_attention_frequency == 0:
            self.cross_attention = QFormerCrossAttention(config)
            self.cross_attention_output = QFormerAttentionOutput(config)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        self.intermediate_query = QFormerIntermediate(config)
        self.output_query = QFormerOutput(config)

    def _feed_forward_chunk_query(self, attention_output: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output

    def forward(
        self,
        query_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sa_output = self.self_attention(query_states, attention_mask=query_attention_mask)
        query_states = self.self_attention_output(sa_output, query_states)

        if self.has_cross_attention:
            ca_output = self.cross_attention(
                query_states=query_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            query_states = self.cross_attention_output(ca_output, query_states)

        query_states = self._feed_forward_chunk_query(query_states)
        return query_states


class SpeechProjector(nn.Module):
    """Projects acoustic features to language model dimension using Q-Former architecture."""

    def __init__(
        self,
        config: SpeechProjectorConfig,
        window_size: Optional[int] = None,
        downsample_rate: Optional[int] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
    ):
        super().__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        self.window_size = window_size if window_size is not None else getattr(config, "window_size", None)
        self.downsample_rate = downsample_rate if downsample_rate is not None else getattr(config, "downsample_rate", None)
        if self.window_size is None or self.downsample_rate is None:
            raise ValueError("window_size and downsample_rate must be provided via args or config.")

        self.num_queries = getattr(config, "num_queries", None)
        if self.num_queries is None:
            self.num_queries = self.window_size // self.downsample_rate

        # Learnable query tokens that aggregate encoder outputs via cross-attention
        self.query_embeds = nn.Parameter(torch.zeros(1, self.num_queries, config.encoder_dim))
        nn.init.normal_(self.query_embeds, mean=0.0, std=1.0)

        self.input_layernorm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)
        self.input_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.layers = nn.ModuleList(
            [QFormerLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        self.output_proj = nn.Linear(config.encoder_dim, config.decoder_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1)

    def _expand_query_embeds(self, num_windows: int, device: torch.device) -> torch.Tensor:
        return self.query_embeds.expand(num_windows, -1, -1).to(device)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = encoder_hidden_states.shape
        device = encoder_hidden_states.device

        # Partition encoder outputs into fixed-size windows for efficient processing
        nblocks = math.ceil(seq_len / self.window_size)
        pad = nblocks * self.window_size - seq_len
        if pad > 0:
            encoder_hidden_states = F.pad(encoder_hidden_states, (0, 0, 0, pad), "constant", 0)

        encoder_hidden_states = encoder_hidden_states.view(batch_size * nblocks, self.window_size, dim)

        query_states = self._expand_query_embeds(batch_size * nblocks, device=device)
        query_states = self.input_layernorm(query_states)
        query_states = self.input_dropout(query_states)

        for layer in self.layers:
            query_states = layer(
                query_states=query_states,
                encoder_hidden_states=encoder_hidden_states,
                query_attention_mask=None,
                encoder_attention_mask=None,
            )

        query_states = query_states.view(batch_size, nblocks * self.num_queries, -1)

        # Project from encoder dimension to decoder dimension for language model input
        projected_states = self.output_proj(query_states)
        return projected_states