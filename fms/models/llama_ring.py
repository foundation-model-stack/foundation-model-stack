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
            # Suggestion 3: Clamp Only Valid Entries
            max_pos = getattr(attn_data.position_encoder, 'max_position_embeddings', 2048)
            position_ids_safe = position_ids.clone()
            position_ids_safe[valid_mask] = position_ids_safe[valid_mask].clamp(0, max_pos - 1)

            # Compute RoPE on the full batch (includes padded values)
            queries_rope, keys_rope = attn_data.position_encoder.adjusted_qk(
                queries, keys, position_ids_safe
            )

            # Suggestion 4: Skip RoPE When No Valid Tokens (Optimized assignment)
            if valid_mask.all():
                queries = queries_rope
                keys = keys_rope
            else:
                # Keep original (unrotated) values at padded positions
                mask_q = valid_mask.unsqueeze(-1).unsqueeze(-1)  # shape [B, T, 1, 1]
                queries = torch.where(mask_q, queries_rope, queries)
                keys    = torch.where(mask_q, keys_rope, keys)

    return (
        queries.permute(0, 2, 1, 3),  # Suggestion 5: Use reshape/permute (transpose is equivalent here) -> Keep transpose for clarity
        keys.permute(0, 2, 1, 3),
        values.permute(0, 2, 1, 3)
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
    # Unpack input if it's a tuple
    input_tensor = x[0] if isinstance(x, tuple) else x

    # Inlined _forward_ring_attention logic
    residual = input_tensor
    x_norm_local = self.ln(input_tensor)

    correct_valid_len = distributed_strategy._local_valid_len

    # Run ring attention forward directly
    x, cache, _ = self.ring_helper.forward(
        x_norm_local,
        mask=mask,
        strategy=distributed_strategy,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state,
        is_causal_mask=is_causal_mask,
        valid_len=correct_valid_len,
        residual=residual,
    )

    return (x, cache) if use_cache else x

