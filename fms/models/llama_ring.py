import logging
import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.distributed as dist # Import dist for rank info
from torch.distributed import P2POp  # For batch_isend_irecv

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
        residual=residual,
        mask=mask,
        strategy=distributed_strategy,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state,
        is_causal_mask=is_causal_mask,
        valid_len=correct_valid_len,
    )

    return (x, cache) if use_cache else x

class RingAttentionHelper:
    def __init__(self, attn_module, strategy, llama_block, use_cache=False, ff=None, ff_norm=None):
        self.attn = attn_module
        self.ff = ff
        self.ff_norm = ff_norm
        self.strategy = strategy
        self.use_cache = use_cache
        self.llama_block = llama_block
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.head_dim = attn_module.emb_kq_per_head
        self.scale = self.head_dim ** 0.5

    def forward(self, x_norm, residual, strategy, mask=None, position_ids=None, past_key_value_state=None,
                is_causal_mask=False, rank=0, valid_len=0):

        start_idx_global = self.rank * self.strategy.block_size
        B, T = x_norm.shape[:2]

        if position_ids is None:
            position_ids = torch.full((B, T), fill_value=-1, dtype=torch.long, device=x_norm.device)
            if valid_len > 0:
                valid_pos = torch.arange(start_idx_global, start_idx_global + valid_len, device=x_norm.device)
                position_ids[:, :valid_len] = valid_pos.unsqueeze(0)

        q_local, k_local, v_local = self.llama_block.compute_local_qkv_and_rope(
            self.attn,
            q=x_norm, k=x_norm, v=x_norm,
            position_ids=position_ids,
            use_cache=False,
            past_key_value_state=past_key_value_state,
            is_self=True
        )

        q_local = q_local[:, :, :valid_len, :]
        k_local = k_local[:, :, :valid_len, :]
        v_local = v_local[:, :, :valid_len, :]

        x_norm_local = x_norm[:, :valid_len, :]
        residual_local = residual[:, :valid_len, :]

        attn_out = self.attention(
            q_local=q_local, 
            k_local=k_local, 
            v_local=v_local, 
            mask_global=mask, 
            valid_len=valid_len, 
            q_start_global=start_idx_global
        )

        result = self.feedforward(
            attn_out=attn_out, 
            residual=residual_local, 
            valid_len=valid_len
        )

        return result, None, None
    
    def attention(self, q_local: torch.Tensor, k_local: torch.Tensor, v_local: torch.Tensor, 
                  mask_global: Optional[torch.Tensor], valid_len, q_start_global):
        """
        Computes ring-style attention using a two-pass approach.
        """
        B, H, T_q_local, D_head = q_local.shape
        D_v = self.attn.emb_v_per_head

        max_score = self._compute_max_score_pass(
            q_local, k_local, mask_global, q_start_global, T_q_local
        )

        numerator, denominator = self._compute_sums_pass(
            q_local, k_local, v_local, mask_global, q_start_global, T_q_local, max_score
        )

        attn_out_h = numerator / (denominator + 1e-10)
        attn_out = attn_out_h.to(q_local.dtype)
        assert attn_out.shape == (B, H, T_q_local, D_v), f"Unexpected attn_out shape: {attn_out.shape}"
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, T_q_local, H * D_v)
        attn_out = self.attn.dense(attn_out)
        return attn_out
    
    def feedforward(self, attn_out, residual, valid_len):
        """
        Applies the feedforward block after attention, including residual connections and padding.
        """
        residual_1 = residual + attn_out
        residual_1_padded = self._pad_to_block(residual_1, self.strategy.block_size, dim=1)
        ff_ln_out_padded = self.ff_norm(residual_1_padded)
        ff_out_padded = self.ff(ff_ln_out_padded)
        ff_out_trimmed = ff_out_padded[:, :valid_len, :]
        
        return ff_out_trimmed + residual_1 
    
    def _compute_max_score_pass(self, q_local: torch.Tensor, k_local: torch.Tensor,
                                mask_global: Optional[torch.Tensor], q_start_global: int,
                                valid_len_local: int) -> torch.Tensor:
        """
        First pass: Computes maximum attention scores for numerical stability.
        """
        B, H, T_q_local, _ = q_local.shape

        device = q_local.device
        dtype = torch.float32
        max_score = torch.full((B, H, T_q_local, 1), -float("inf"), device=device, dtype=dtype)
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)
        current_k_block = k_local
        current_k_len = k_local.shape[2]

        for i in range(self.world_size):
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.strategy.block_size
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)

            current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len] if mask_global is not None else None
            if current_k_len == 0:
                if i < self.world_size - 1:
                    current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                continue
            
            scores = self._compute_attention_scores(
                q_local, current_k_block[:, :, :current_k_len, :],
                q_indices_global, k_indices_global, current_mask, apply_mask=True
            )
            
            max_score = self._update_max_score(scores, max_score)
            if i < self.world_size - 1:
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)

        return max_score


    def _compute_sums_pass(self, q_local: torch.Tensor, k_local: torch.Tensor, v_local: torch.Tensor,
                           mask_global: Optional[torch.Tensor], q_start_global: int,
                           valid_len_local: int, max_score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Second pass: Computes numerator and denominator for the softmax.
        """

        B, H, T_q_local, _ = q_local.shape

        D_v = self.attn.emb_v_per_head
        device = q_local.device
        dtype = torch.float32
        numerator = torch.zeros(B, H, T_q_local, D_v, device=device, dtype=dtype)
        denominator = torch.zeros(B, H, T_q_local, 1, device=device, dtype=dtype)
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)
        current_k_block = k_local
        current_v_block = v_local
        current_k_len = k_local.shape[2]

        for i in range(self.world_size):
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.strategy.block_size
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)
            if current_k_len == 0:
                if i < self.world_size - 1:
                    current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                    current_v_block, _ = self._ring_shift_tensor(current_v_block, self.strategy.block_size)
                continue
            current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len] if mask_global is not None else None
            scores = self._compute_attention_scores(
                q_local, current_k_block[:, :, :current_k_len, :],
                q_indices_global, k_indices_global, current_mask, apply_mask=True
            )
            numerator, denominator = self._update_totals(
                scores, current_v_block[:, :, :current_k_len, :], max_score, numerator, denominator
            )
            
            if i < self.world_size - 1:
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                current_v_block, _ = self._ring_shift_tensor(current_v_block, self.strategy.block_size)

        return numerator, denominator

    def _compute_attention_scores(self, q, k, q_indices_global, k_indices_global, mask=None,
                                  apply_mask=True, keep_causal=True):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if apply_mask and mask is not None:
            scores = scores + mask.to(scores.dtype)
        if apply_mask and keep_causal:
            causal_mask = (k_indices_global[None, :] > q_indices_global[:, None]).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, -torch.inf)

        return scores

    def _update_max_score(self, scores, current_max):
        block_max = scores.masked_fill(scores == -torch.inf, torch.finfo(scores.dtype).min).amax(dim=-1, keepdim=True)
        return torch.maximum(current_max, block_max)

    def _update_totals(self, scores, v, max_score, numerator, denominator):
        stable_scores = (scores - max_score).clamp(min=-10.0, max=10.0)
        exp_scores = torch.exp(stable_scores)

        # Handle potential NaNs or Infs in exp_scores after masking/clamping
        exp_scores = torch.nan_to_num(exp_scores, nan=0.0, posinf=torch.finfo(exp_scores.dtype).max, neginf=0.0)

        numerator += torch.einsum("bhqk,bhkd->bhqd", exp_scores, v.to(numerator.dtype))
        denominator += exp_scores.sum(dim=-1, keepdim=True)
        return numerator, denominator

    def _ring_shift_tensor(self, tensor: torch.Tensor, pad_len: int) -> Tuple[torch.Tensor, int]:
        """
        Ring‑shifts `tensor` along its last dimension by one rank:
        - GPU: all_gather of lengths + all_to_all of only the neighbor's block.
        Returns (received_tensor cropped to true length, received_length).
        """

        rank, world = self.rank, self.world_size
        send_rank = (rank + 1) % world
        recv_rank = (rank - 1 + world) % world
        valid_len = tensor.shape[-2]

        padded = self._pad_to_block(tensor, pad_len, dim=-2).contiguous()
        device = tensor.device

        # Create buffers
        send_len = torch.tensor([valid_len], dtype=torch.int32, device=device)
        recv_len = torch.empty(1, dtype=torch.int32, device=device)
        tensor_recv = torch.empty_like(padded)

        # Prepare send/recv ops
        print(f"DEBUG: rank: {self.rank}, padded.size():{padded.size()}, send_len: {send_len}")
        ops = [
            P2POp(op=dist.isend, tensor=send_len, peer=send_rank),
            P2POp(op=dist.irecv, tensor=recv_len, peer=recv_rank),
            P2POp(op=dist.isend, tensor=padded, peer=send_rank),
            P2POp(op=dist.irecv, tensor=tensor_recv, peer=recv_rank),
        ]
        
        # Execute async, then wait
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        recv_len = recv_len.item()
        return tensor_recv, recv_len
    
    @staticmethod
    def _pad_to_block(t, target_len, dim=2):
        pad_len = target_len - t.shape[dim]
        if pad_len <= 0:
            return t
        pad_shape = list(t.shape)
        pad_shape[dim] = pad_len
        pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=dim)

