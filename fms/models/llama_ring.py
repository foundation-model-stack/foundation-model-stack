import logging
import re
from dataclasses import dataclass
import time
from typing import Any, List, Mapping, Optional, Tuple

from fms.models.ring_attention_helper import RingAttentionHelper
import torch
import torch.nn as nn
import torch.distributed as dist # Import dist for rank info

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
    RingAttentionStrategy,
    TensorParallelStrategy,
)
from fms.modules.attention import MultiHeadAttention
from fms.modules.embedding import WordEmbedding
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)




# class RingAttentionBase:

def compute_local_qkv_and_rope(
    self, attn_data, q, k=None, v=None, position_ids=None,
    use_cache=False, past_key_value_state=None, is_self=True
):
    B, T, _ = q.shape
    q_out, k_out, v_out = attn_data.in_proj(q, k, v)

    queries = q_out.view(B, T, attn_data.nheads, attn_data.emb_kq_per_head)
    keys    = k_out.view(B, T, attn_data.kvheads, attn_data.emb_kq_per_head)
    values  = v_out.view(B, T, attn_data.kvheads, attn_data.emb_v_per_head)

    if attn_data.position_encoder is not None and T > 0:
        assert position_ids is not None, "position_ids must be provided for rotary encoding"
        # Adjust assertion for T=0 case where position_ids might be (B, 0)
        expected_pos_shape = (B, T)
        if not (T == 0 and position_ids.shape == (B, 0)) and position_ids.shape != expected_pos_shape:
                raise AssertionError(f"Expected position_ids shape {expected_pos_shape}, got {position_ids.shape}")

        # Identify valid tokens
        valid_mask = position_ids != -1
        if valid_mask.any():
            # Clamp safe indexing into RoPE cache (even though -1s are masked)
            max_pos = getattr(attn_data.position_encoder, 'max_position_embeddings', 2048)
            position_ids_safe = position_ids.clamp(min=0, max=max_pos - 1)

            # Compute RoPE on the full batch (includes padded values)
            queries_rope, keys_rope = attn_data.position_encoder.adjusted_qk(
                queries, keys, position_ids_safe
            )

            # Keep original (unrotated) values at padded positions
            mask_q = valid_mask.unsqueeze(-1).unsqueeze(-1)  # shape [B, T, 1, 1]
            queries = torch.where(mask_q, queries_rope, queries)
            keys    = torch.where(mask_q, keys_rope, keys)

    return (
        queries.transpose(1, 2),  # [B, H, T, D]
        keys.transpose(1, 2),
        values.transpose(1, 2)
    )


def forward_ring(
    self,
    x,
    *,
    mask=None,
    position_ids=None,
    past_key_value_state=None,
    use_cache=False,
    is_causal_mask=False,
    attn_algorithm=None,
    distributed_strategy: Optional[DistributedStrategy] = None,
):
    if isinstance(distributed_strategy, RingAttentionStrategy):
        rank = dist.get_rank()
        x, cache, _ = self._forward_ring_attention(
            x,
            mask=mask,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            is_causal_mask=is_causal_mask,
            strategy=distributed_strategy,
            verbosity=0,
            rank=rank,
        )
    else:
        x, cache, _ = self._forward_engine_attention(
            x,
            mask=mask,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            is_causal_mask=is_causal_mask,
            verbosity=0,
        )

    return (x, cache) if use_cache else x


def _forward_ring_attention(
    self,
    x,
    *,
    mask,
    position_ids,
    past_key_value_state,
    use_cache,
    is_causal_mask,
    strategy,
    verbosity: int,
    rank=0,
):
    residual = x
    x_norm_local = self.ln(x)
    ring_helper = RingAttentionHelper(
        attn_module=self.attn,
        strategy=strategy,
        llama_block=self,
        use_cache=use_cache,
        ff=self.ff_sub_layer,
        ff_norm=self.ff_ln,
    )
    correct_valid_len = strategy._local_valid_len
    x, cache, _ = ring_helper.forward(
        x_norm_local,
        mask=mask,
        strategy=strategy,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state,
        is_causal_mask=is_causal_mask,
        rank=rank,
        valid_len=correct_valid_len,
        residual=residual,
    )
    return x, cache, None
