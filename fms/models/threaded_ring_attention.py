import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union
import threading
import queue

import torch
import torch.nn as nn
from torch import Tensor, nn

from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import GatedLinearUnit

logger = logging.getLogger(__name__)

@dataclass
class BlockData:
    engine_instance: 'ThreadedRingAttentionEngine' 
    block_id: int
    num_blocks: int
    q_start: int
    q_end: int
    mask_global: Optional[Tensor]
    block_queues: List[queue.Queue]
    await_max: threading.Barrier
    await_sums: threading.Barrier
    result_buffer: Dict[int, Tensor]
    q_block: Tensor
    k_local: Tensor
    v_local: Tensor
    x_block: Tensor
    x_norm_block: Tensor # Add x_norm block



# fake ring attention, for now (no multi gpu setup quite yet)
class ThreadedRingAttentionEngine:

    def __init__(self, block_size: int, attn: MultiHeadAttention, ff: GatedLinearUnit, ff_norm: nn.Module, is_causal: bool):

        self.block_size = block_size
        self.attn = attn
        self.ff = ff
        self.ff_norm = ff_norm
        self.is_causal = is_causal
        self.head_dim = attn.emb_kq_per_head
        self.scale = math.sqrt(self.head_dim)


    # main method
    def forward_full(self, q_global: Tensor, k_global: Tensor, v_global: Tensor, mask_global: Optional[Tensor], x_global: Tensor, x_norm_global: Tensor) -> Tensor:

        T_q = q_global.shape[2]

        q_starts = list(range(0, T_q, self.block_size))
        num_blocks = len(q_starts) # Renamed from num_threads
        if num_blocks == 0: return torch.empty_like(x_global)

        result_buffer: Dict[int, Tensor] = {}

        block_queues = [queue.Queue() for _ in range(num_blocks)] # Renamed
        max_barrier = threading.Barrier(num_blocks) # Renamed
        sum_barrier = threading.Barrier(num_blocks) # Renamed

        threads = []
        for block_id, q_start in enumerate(q_starts):
            q_end = min(q_start + self.block_size, T_q)
            q_block, k_block, v_block, x_block = (
                q_global[:, :, q_start:q_end, :],
                k_global[:, :, q_start:q_end, :],
                v_global[:, :, q_start:q_end, :],
                x_global[:, q_start:q_end, :]
            )
            # Slice the global x_norm for this block
            x_norm_block = x_norm_global[:, q_start:q_end, :]

            block_data = BlockData(
                engine_instance=self, block_id=block_id, num_blocks=num_blocks, q_start=q_start, q_end=q_end, mask_global=mask_global, block_queues=block_queues, 
                await_max=max_barrier, await_sums=sum_barrier, result_buffer=result_buffer,
                q_block=q_block, k_local=k_block, v_local=v_block, x_block=x_block, x_norm_block=x_norm_block # Pass x_norm_block
            )

            thread = threading.Thread(target=ThreadedRingAttentionEngine.block_worker, args=(block_data,), daemon=True)
            threads.append(thread)
            thread.start()

        for thread in threads: thread.join()

        ordered_results = [result_buffer[q_start] for q_start in q_starts]
        final_output = torch.cat(ordered_results, dim=1)

            return final_output
    
    # block outline
    @staticmethod
    def block_worker(args: BlockData):

        engine = args.engine_instance
        initial_max_score, initial_num, initial_den = engine.init_values(args.q_block)

        final_max_score = engine.compute_max_score(args, initial_max_score)
        args.await_max.wait()

        final_num, final_den = engine.compute_sums(args, final_max_score, initial_num, initial_den)
        args.await_sums.wait()

        # Separate computation steps for clarity (previously for logging)
        attn_out_raw, residual_1 = engine.compute_attn_out_and_residual1(args.x_block, final_num, final_den)
        ff_ln_out = engine.ff_norm(residual_1)
        ff_out_raw = engine.ff(ff_ln_out)
        block_output = residual_1 + ff_out_raw # Assuming no dropout in engine

        args.result_buffer[args.q_start] = block_output

    """ compute max scores for stability (first flash pass) """
    def compute_max_score(self, args: BlockData, initial_max_score: Tensor) -> Tensor:

        engine = args.engine_instance

        next_block_id = (args.block_id + 1) % args.num_blocks
        send_q, recv_q = args.block_queues[next_block_id], args.block_queues[args.block_id]

        device = args.q_block.device
        q_indices = torch.arange(args.q_start, args.q_end, device=device)
        local_k_indices = torch.arange(args.q_start, args.q_end, device=device)

        max_score = initial_max_score
        current_k, current_k_idx, current_k_global_start = args.k_local, local_k_indices, args.q_start

        for i in range(args.num_blocks):
            mask = args.mask_global[:, :, args.q_start:args.q_end, current_k_global_start:current_k_global_start+current_k.shape[2]] if args.mask_global is not None else None

            # Call helper directly (no debug logging)
            max_score = engine.update_max_attn(args.q_block, current_k, mask, q_indices, current_k_idx, max_score)
            if i < args.num_blocks - 1:
                send_q.put((current_k, current_k_idx, current_k_global_start))
                current_k, current_k_idx, current_k_global_start = recv_q.get()

        return max_score

    """ sum loop """
    def compute_sums(self, args: BlockData, final_max_score: Tensor, initial_num: Tensor, initial_den: Tensor) -> Tuple[Tensor, Tensor]:

        engine = args.engine_instance
        prev_block_id = (args.block_id - 1 + args.num_blocks) % args.num_blocks
        next_block_id = (args.block_id + 1) % args.num_blocks
        send_q, recv_q = args.block_queues[next_block_id], args.block_queues[args.block_id]

        device = args.q_block.device
        q_indices = torch.arange(args.q_start, args.q_end, device=device)
        local_k_indices = torch.arange(args.q_start, args.q_end, device=device)

        num, den = initial_num, initial_den
        current_k, current_v, current_k_idx, current_k_global_start = args.k_local, args.v_local, local_k_indices, args.q_start

        for i in range(args.num_blocks):
            mask = args.mask_global[:, :, args.q_start:args.q_end, current_k_global_start:current_k_global_start+current_k.shape[2]] if args.mask_global is not None else None

            # Call helper directly (no debug logging)
            num, den = engine.update_totals(args.q_block, current_k, current_v, mask, q_indices, current_k_idx, final_max_score, num, den)
            if i < args.num_blocks - 1:
                send_q.put((current_k, current_v, current_k_idx, current_k_global_start))
                current_k, current_v, current_k_idx, current_k_global_start = recv_q.get()

        return num, den
    
    """ final output """
    def compute_attn_out_and_residual1(self, x_residual: Tensor, num: Tensor, den: Tensor) -> Tuple[Tensor, Tensor]:

        B, q_len, E = x_residual.shape; H, D_v = num.shape[1], num.shape[3]
        attn_out_h = num / (den + 1e-10) # Added epsilon for stability
        attn_out = attn_out_h.transpose(1, 2).contiguous().view(B, q_len, H * D_v)
        attn_out = self.attn.dense(attn_out)
        residual_1 = x_residual + attn_out
        return attn_out, residual_1

    """ Helper Functions"""

    # Simplified raw_attention, assuming masking is always applied when needed
    def raw_attention(self, q: Tensor, k: Tensor, mask: Optional[Tensor], q_indices: Tensor, k_indices: Tensor) -> Tensor:

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / self.scale
        if mask is not None: scores = scores + mask
        if self.is_causal:
            q_indices_dev = q_indices.to(k_indices.device)
            causal_mask = (k_indices[None, :] > q_indices_dev[:, None]).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, -torch.inf)
        return scores

    def init_values(self, q: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        B, H, q_len, D_head = q.shape; device, dtype = q.device, q.dtype
        D_v = self.attn.emb_v_per_head

        max_score = torch.full((B, H, q_len, 1), -torch.inf, dtype=torch.float32, device=device) # Use float32 for stability
        numerator = torch.zeros(B, H, q_len, D_v, dtype=torch.float32, device=device) # Use float32 for stability
        denominator = torch.zeros(B, H, q_len, 1, dtype=torch.float32, device=device) # Use float32 for stability

        return max_score, numerator, denominator

    # Removed args and step_idx params, simplified signature
    def update_max_attn(self, q: Tensor, k: Tensor, mask: Optional[Tensor], q_indices: Tensor, k_indices: Tensor, current_max_score: Tensor) -> Tensor:

        # Compute scores WITH mask for actual calculation
        attn_scores = self.raw_attention(q, k, mask, q_indices, k_indices)

        block_max = attn_scores.masked_fill(attn_scores == -torch.inf, torch.finfo(attn_scores.dtype).min).amax(dim=-1, keepdim=True)
        return torch.maximum(current_max_score, block_max)

    # Removed args and step_idx params, simplified signature
    def update_totals(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor], q_indices: Tensor, k_indices: Tensor, final_max_score: Tensor, current_num: Tensor, current_den: Tensor) -> Tuple[Tensor, Tensor]:
        attn_scores = self.raw_attention(q, k, mask, q_indices, k_indices)

        stable_scores = (attn_scores - final_max_score).clamp(min=-10.0, max=10.0) # Clamping for stability
        exp_scores = torch.exp(stable_scores)
        num_update = torch.einsum("bhqk,bhkd->bhqd", exp_scores, v.to(current_num.dtype)) # Ensure v matches accumulator dtype
        den_update = exp_scores.sum(dim=-1, keepdim=True)
        return current_num + num_update, current_den + den_update



  