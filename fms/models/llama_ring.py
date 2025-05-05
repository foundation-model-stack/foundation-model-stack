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
    x, # This 'x' might be a tuple (tensor, cache_state) from the previous layer if use_cache=True
    *,
    mask=None,
    position_ids=None,
    past_key_value_state=None,
    use_cache=False,
    is_causal_mask=False,
    attn_algorithm=None, # Keep standard attn args
    distributed_strategy: Optional[DistributedStrategy] = None,
):
    # If x is a tuple (result of previous layer with use_cache=True), unpack it
    input_tensor = x[0] if isinstance(x, tuple) else x
    # Note: We might need the past_key_value_state from x[1] if applicable,
    # but RingAttention doesn't use it internally yet. The 'past_key_value_state'
    # argument passed to this function is likely the one intended anyway.

    if isinstance(distributed_strategy, RingAttentionStrategy):
        # --- RING ATTENTION PATH ---
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Call _forward_ring_attention with the actual tensor
        output_ring = self._forward_ring_attention(
            input_tensor, # Pass the unpacked tensor
            mask=mask,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state, # Ring doesn't use this internally
            use_cache=use_cache, # Pass the original flag to determine return structure
            is_causal_mask=is_causal_mask,
            strategy=distributed_strategy,
            rank=rank,
        )
        # _forward_ring_attention returns x_output or (x_output, None) based on use_cache
        return output_ring # Return whatever _forward_ring_attention returned

    else:
        # --- STANDARD ATTENTION PATH (Non-Ring) ---
        # Standard path expects the tensor input directly
        self_attn_past_key_value = past_key_value_state

        # MHA and Add&Norm
        residual = input_tensor # Use the unpacked tensor
        x_norm = self.ln(input_tensor) # Use the unpacked tensor
        attn_output = self.attn(
            q=x_norm,
            mask=mask,
            position_ids=position_ids,
            attn_algorithm=attn_algorithm,
            past_key_value_state=self_attn_past_key_value,
            use_cache=use_cache,
            is_self=True,
            is_causal_mask=is_causal_mask,
        )
        cache = None
        if use_cache:
            x_attn, cache = attn_output
        else:
            x_attn = attn_output

        if hasattr(self, 'dropout') and self.config.p_dropout != 0:
            x_attn = self.dropout(x_attn)
        # Residual connection
        x = x_attn + residual # Use residual derived from input_tensor

        # FF and Add&Norm
        residual = x # Use the result from the attention block
        x_ff = self.ff_ln(x) # Use the result from the attention block
        x_ff = self.ff_sub_layer(x_ff)
        if hasattr(self, 'dropout') and self.config.p_dropout != 0:
            x_ff = self.dropout(x_ff)
        # Another residual
        x = x_ff + residual # Use residual from the attention block

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
    strategy, # Keep strategy
    rank=0,
):
    residual = x
    
    # Apply LayerNorm locally
    x_norm_local = self.ln(x) 

    ring_helper = RingAttentionHelper(
        attn_module=self.attn,
        strategy=strategy,
        llama_block=self,
        use_cache=False, # Ring helper doesn't support cache internally
        ff=self.ff_sub_layer,              # Match engine args (though engine is removed)
        ff_norm=self.ff_ln,                # Match engine args (though engine is removed)
    )

    correct_valid_len = strategy._local_valid_len # Use the strategy's valid length

    # Call the helper's forward, which now returns only the tensor output
    x_output = ring_helper.forward(
        x_norm_local, # Pass the local normalized tensor
        mask=mask,
        strategy=strategy,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state, # Usually None for ring?
        is_causal_mask=is_causal_mask,
        rank=rank,
        valid_len=correct_valid_len, # Pass the correct valid length
        residual=residual # Pass the local residual source
    )

    # Cache handling: Ring attention path currently doesn't manage KV cache
    # If caching is needed with RingAttention, it would require adjustments
    # potentially outside the ring_helper itself.
    cache = None # Explicitly set cache to None for ring path

    # Return structure depends on use_cache flag passed to the main forward method
    if use_cache:
        # Even though ring doesn't produce a cache, return None to match signature
        return x_output, cache
    else:
        return x_output