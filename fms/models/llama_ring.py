import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from fms.modules.attention import MultiHeadAttention
from fms.distributed.strategy import DistributedStrategy, RingAttentionStrategy



def ring_forward(
    self,
    x,
    *,
    mask=None,
    position_ids=None,
    past_key_value_state=None,
    use_cache=False,
    is_causal_mask=False,
    attn_algorithm=None
):
    
    residual = x 
    x_norm = self.ln(x)


    x = RingAttentionKernel.ring_attention(
        x_norm=x_norm,
        attn_module=self.attn,
        strategy=self.distributed_strategy,
        valid_len=self.distributed_strategy._local_valid_len,
        mask=mask, 
        position_ids=position_ids, # Sharded position_ids
        past_key_value_state=past_key_value_state, 
        causal=is_causal_mask,
    )
    
    # use cache and dropout have not yet been implemented / tested
    x = x + residual

    # then we do FF and Add&Norm
    residual = x
    x = self.ff_ln(x)
    x = self.ff_sub_layer(x)
    x = x + residual

    if use_cache:
        return (x, None)
    else:
        return x

class RingAttentionKernel:

    @staticmethod
    def ring_attention(
        x_norm: Tensor,
        attn_module: MultiHeadAttention,
        strategy: RingAttentionStrategy,
        valid_len: int,
        *,
        mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
        causal: bool = False,
    ) -> Tensor:
        
        batch_size, num_valid_tokens_input_shard, emb_dim = x_norm.shape 
        assert num_valid_tokens_input_shard == valid_len
        current_rank_token_global_start_idx = strategy.rank * strategy.block_size

        # slice to valid length to be safe
        current_rank_input_slice = x_norm[:, :valid_len]

        # compute position ids
        if position_ids is not None:
            position_ids_for_rope_computation = position_ids[:, current_rank_token_global_start_idx : current_rank_token_global_start_idx + valid_len]
        elif valid_len > 0:
            position_ids_for_rope_computation = torch.arange(current_rank_token_global_start_idx, current_rank_token_global_start_idx + valid_len, device=x_norm.device).unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids_for_rope_computation = None

        # compute QKV + RoPE
        if valid_len:
            q, k, v = RingAttentionKernel._compute_qkv_and_rope(
                attn_module, current_rank_input_slice, position_ids_for_rope_computation
            )
        else:
            nheads, emb_kq_per_head, emb_v_per_head = attn_module.nheads, attn_module.emb_kq_per_head, attn_module.emb_v_per_head
            q = k = torch.empty((batch_size, nheads, 0, emb_kq_per_head), device=x_norm.device, dtype=x_norm.dtype)
            v = torch.empty((batch_size, nheads, 0, emb_v_per_head), device=x_norm.device, dtype=x_norm.dtype)

        scale = attn_module.scale_factor or math.sqrt(attn_module.emb_kq_per_head)
        
        accum_dtype = torch.float32

        # main ring attention 
        out = RingAttentionKernel._compute_attention_ring(
            q, k, v, mask, strategy, current_rank_token_global_start_idx, valid_len, scale, accum_dtype, causal # valid_len is num_valid_tokens for this rank
        )

        if valid_len:
            proj = out.transpose(1, 2).reshape(batch_size, valid_len, -1)
            out = attn_module.dense(proj)
        else:
            out = torch.empty((batch_size, 0, emb_dim), device=x_norm.device, dtype=x_norm.dtype)

        return out


    @staticmethod
    def _compute_qkv_and_rope(
        attn: MultiHeadAttention,
        x: Tensor,
        rope_position_ids: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, seq_len, _ = x.shape # x is current_rank_input_slice, so seq_len is valid_len for this rank
        q_proj, k_proj, v_proj = attn.in_proj(x, None, None)
        nheads, kvheads = attn.nheads, attn.kvheads
        emb_kq_per_head, emb_v_per_head = attn.emb_kq_per_head, attn.emb_v_per_head

        # reshape & apply RoPE if needed
        q = q_proj.view(batch_size, seq_len, nheads, emb_kq_per_head)
        k = k_proj.view(batch_size, seq_len, kvheads, emb_kq_per_head)
        v = v_proj.view(batch_size, seq_len, kvheads, emb_v_per_head)
        if attn.position_encoder and seq_len:
            assert rope_position_ids is not None
            valid_rope_pos_mask = rope_position_ids.ne(-1)
            if valid_rope_pos_mask.any():
                rope_internal_max_seq_len = getattr(attn.position_encoder, "max_seq_len", 2048)
                clamped_rope_ids = rope_position_ids.clamp(0, rope_internal_max_seq_len - 1)
                q, k = attn.position_encoder.adjusted_qk(q, k, clamped_rope_ids)

        q, k, v = [x_tensor.permute(0, 2, 1, 3) for x_tensor in (q, k, v)]
        if nheads != kvheads:
            kv_to_q_head_ratio = nheads // kvheads
            k = k.repeat_interleave(kv_to_q_head_ratio, dim=1)
            v = v.repeat_interleave(kv_to_q_head_ratio, dim=1)
        return q, k, v
    
    @staticmethod
    def _compute_attention_ring(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        strategy: RingAttentionStrategy,
        q_start: int, # global start index for queries in q
        num_valid_tokens: int,   # number of queries in q for this rank's block (num_queries_in_block)
        scale: float,
        accum_dtype: torch.dtype,
        causal: bool,
    ) -> Tensor:
        
        # compute max score for normalization 
        # this could be optimized with online softmax
        max_score = RingAttentionKernel._max_pass(
            q, k, mask, q_start, num_valid_tokens, strategy, scale, causal, accum_dtype
        )

        # compute the numerator and denominator for attention calc
        numerator, denominator = RingAttentionKernel._sum_pass(
            q, k, v, mask, q_start, num_valid_tokens, max_score, strategy, scale, accum_dtype, causal
        )

        # guard against empty calc
        if num_valid_tokens == 0:
            return torch.empty((q.shape[0], q.shape[1], 0, v.shape[-1]),
                               device=q.device, dtype=q.dtype)
        return (numerator / (denominator + torch.finfo(denominator.dtype).eps)).to(q.dtype)
    

    @staticmethod
    def _attn_scores(
        Q: Tensor,
        K: Tensor,
        query_indices: Tensor, # global indices for queries in Q
        key_indices: Tensor,   # global indices for keys in K
        scale: float,
        mask: Optional[Tensor],
        causal: bool,
    ) -> Tensor:
        batch_size, nheads, num_q, _ = Q.shape # num_q is num_queries_in_block for Q
        num_k = K.shape[2]          # num_k is current_block_k_len for K
        if num_q == 0 or num_k == 0:
            return Q.new_empty((batch_size, nheads, num_q, num_k))

        scores = torch.matmul(Q / scale, K.transpose(-2, -1))
        if mask is not None:
            scores = scores + mask.to(scores.dtype)
        if causal:
            # build a [1,1,q_len,k_len] mask where key_pos > query_pos
            future_mask = (key_indices[None, :] > query_indices[:, None])
            future_mask = future_mask.unsqueeze(0).unsqueeze(0) 
            scores = scores.masked_fill(future_mask, float("-inf"))
        return scores
    
    @staticmethod
    def _max_pass(
        q: Tensor,
        k: Tensor,
        mask: Optional[Tensor],
        q_start: int, # global start index for queries in q
        num_valid_tokens: int,   # number of queries in q for this rank's block
        strategy: RingAttentionStrategy,
        scale: float,
        causal: bool,
        accum_dtype: torch.dtype,
    ) -> Tensor:
        batch_size, nheads, _, _ = q.shape
        max_score = torch.full((batch_size, nheads, num_valid_tokens, 1),
                               torch.finfo(accum_dtype).min,
                               device=q.device, dtype=accum_dtype)
        query_global_indices = torch.arange(q_start, q_start + num_valid_tokens, device=q.device)
        q_fp32 = q.to(accum_dtype)
        k_fp32 = k.to(accum_dtype)

        for i in range(strategy.world_size):
            source_rank = (strategy.rank - i) % strategy.world_size
            block_offset_for_source_rank = source_rank * strategy.block_size
            k_len_current_block = k_fp32.shape[2]
            if num_valid_tokens and k_len_current_block:
                key_block_global_indices = torch.arange(block_offset_for_source_rank, block_offset_for_source_rank + k_len_current_block, device=q.device)
                current_attention_mask_slice = (mask[..., q_start:q_start+num_valid_tokens, block_offset_for_source_rank:block_offset_for_source_rank+k_len_current_block]
                        if mask is not None else None)
                current_scores = RingAttentionKernel._attn_scores(q_fp32, k_fp32, query_global_indices, key_block_global_indices, scale, current_attention_mask_slice, causal)
                max_score = torch.maximum(max_score, current_scores.amax(dim=-1, keepdim=True))

            # no need for last round communication
            if i < strategy.world_size - 1:

                # ring attention communication -- shift kvs
                k_fp32, _ = strategy._ring_shift_tensor(k_fp32, k_len_current_block)

        return max_score

    @staticmethod
    def _sum_pass(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor],
        q_start: int, # global start index for queries in q
        num_valid_tokens: int,   # number of queries in q for this rank's block
        max_score: Tensor,
        strategy: RingAttentionStrategy,
        scale: float,
        accum_dtype: torch.dtype,
        causal: bool,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, nheads, _, _ = q.shape
        emb_v_per_head = v.shape[-1]
        numerator = torch.zeros((batch_size, nheads, num_valid_tokens, emb_v_per_head), device=q.device, dtype=accum_dtype)
        denomminator = torch.zeros((batch_size, nheads, num_valid_tokens, 1), device=q.device, dtype=accum_dtype)
        query_global_indices = torch.arange(q_start, q_start + num_valid_tokens, device=q.device)
        q_fp32 = q.to(accum_dtype)
        k_fp32 = k.to(accum_dtype)
        v_fp32 = v.to(accum_dtype)
        
        log_min_exp_threshold = math.log(torch.finfo(accum_dtype).tiny) + 1.0
        log_max_exp_threshold = math.log(torch.finfo(accum_dtype).max) - 1.0

        for i in range(strategy.world_size):
            source_rank = (strategy.rank - i) % strategy.world_size
            block_offset_for_source_rank = source_rank * strategy.block_size
            k_len_current_block = k_fp32.shape[2]
            if num_valid_tokens and k_len_current_block:
                key_block_global_indices = torch.arange(block_offset_for_source_rank, block_offset_for_source_rank + k_len_current_block, device=q.device)
                current_attention_mask_slice = (mask[..., q_start:q_start+num_valid_tokens, block_offset_for_source_rank:block_offset_for_source_rank+k_len_current_block]
                        if mask is not None else None)
                current_scores = RingAttentionKernel._attn_scores(q_fp32, k_fp32, query_global_indices, key_block_global_indices, scale, current_attention_mask_slice, causal)
                score_delta = torch.where(torch.isneginf(max_score), float("-inf"), current_scores - max_score)
                exp_scores = torch.exp(score_delta.clamp(min=log_min_exp_threshold, max=log_max_exp_threshold))
                # exp_scores = exp_scores.masked_fill(torch.isneginf(max_score), 0.0) # This line is likely redundant
                numerator += torch.matmul(exp_scores, v_fp32.narrow(2, 0, k_len_current_block))
                denomminator += exp_scores.sum(dim=-1, keepdim=True)
            
            # no need for last round communication
            if i < strategy.world_size - 1:

                # ring attention communication -- shift kvs
                k_fp32, _ = strategy._ring_shift_tensor(k_fp32, k_len_current_block)
                v_fp32, _ = strategy._ring_shift_tensor(v_fp32, k_len_current_block)

        return numerator, denomminator
    
