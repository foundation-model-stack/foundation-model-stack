import logging
import re
from dataclasses import dataclass
import time
from typing import Any, List, Mapping, Optional, Tuple

from fms.models.ring_attention_helper import RingAttentionHelper
from fms.models.threaded_ring_attention import ThreadedRingAttentionEngine
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




def compute_qkv_and_rope_thread(self, attn_data, q, k=None, v=None, position_ids=None, use_cache=False, past_key_value_state=None, is_self=True):
    B, T, _ = q.shape
    q_out, k_out, v_out = attn_data.in_proj(q, k, v)

    queries = q_out.view(B, T, attn_data.nheads, attn_data.emb_kq_per_head)
    keys = k_out.view(B, T, attn_data.kvheads, attn_data.emb_kq_per_head)
    values = v_out.view(B, T, attn_data.kvheads, attn_data.emb_v_per_head)

    if attn_data.position_encoder is not None:
        if position_ids is None:
            position_ids = torch.arange(T, device=q.device).unsqueeze(0).expand(B, -1)
        queries, keys = attn_data.position_encoder.adjusted_qk(queries, keys, position_ids, past_key_value_state, use_cache)

    return queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
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
    
    # print(x.shape)
    # --- Debug Verbosity Control ---
    # 0: Off
    # 1: Minimal diff summary (norms)
    # 2: Level 1 + Missing keys
    # 3: Level 2 + Detailed diff values
    # 4: Level 3 + Enable internal helper debug logs
    debug_verbosity = 0 # Set desired level here (0-4)
    debug_info = {} if debug_verbosity > 0 else None # Init debug dict only if verbosity > 0
    x_original = x # Store the original input for debug comparison

    if isinstance(distributed_strategy, RingAttentionStrategy):
        # --- RING ATTENTION PATH ---
        rank = dist.get_rank() 
        output_ring = self._forward_ring_attention(
            x,
            mask=mask,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            is_causal_mask=is_causal_mask,
            strategy=distributed_strategy, # Pass strategy
            verbosity=debug_verbosity, # Pass verbosity level
            rank=rank,
        )

        if use_cache:
            x, cache, debug_ring = output_ring
        else:
            x, _, debug_ring = output_ring
            cache = None

        # No nested dict expected — flatten directly
        if debug_verbosity > 0 and debug_ring:
            for k, v in debug_ring.items(): # Keys already have rank suffix from _forward_ring_attention
                debug_info[k] = v # Directly assign without adding another suffix
        
        # --- ENGINE PATH FOR COMPARISON ---
        if debug_verbosity > 0:
            # Gather the ORIGINAL block input for the engine comparison path
            # Pad x_original to the block size before gathering, similar to how
            # RingAttentionStrategy pads tensors before communication.
            padded_x_original = x_original
            target_len = distributed_strategy.block_size
            current_len = x_original.shape[1]
            pad_len = target_len - current_len
            if pad_len > 0:
                pad_shape = list(x_original.shape)
                pad_shape[1] = pad_len
                pad = torch.zeros(*pad_shape, dtype=x_original.dtype, device=x_original.device)
                padded_x_original = torch.cat([x_original, pad], dim=1)
            x_gathered_original = distributed_strategy.gather_output(padded_x_original)
            rank = dist.get_rank() if dist.is_initialized() else 0
            output_engine_gathered = self._forward_engine_attention(
                x_gathered_original, # Pass the gathered original input
                mask=mask,
                position_ids=position_ids,
                past_key_value_state=past_key_value_state,
                use_cache=False,
                is_causal_mask=is_causal_mask,
                verbosity=debug_verbosity, # Pass verbosity level
            )
            # Log the gathered input that was actually passed (optional, but good for clarity)
            # Note: _forward_engine_attention already logs this internally as x_input_r{rank}
            # debug_info[f"engine_x_gathered_input_r{rank}"] = x_gathered_original.clone().detach().cpu()

            _, _, debug_engine = output_engine_gathered

            if debug_engine:
                for k, v in debug_engine.items(): # Keys already have prefix and rank suffix from _forward_engine_attention
                    debug_info[k] = v # Directly assign

            # --- DEBUG DIFF ---
            diffs = self._diff_debug_dicts(
                {k: v for k, v in debug_info.items() if k.startswith("ring")},
                {k: v for k, v in debug_info.items() if k.startswith("engine")},
                debug_verbosity=debug_verbosity # Pass verbosity level
            )
            # print(f"--- Exiting after debug diff in Rank {rank}, Layer {self.layer_index} ---") # Commented out to prevent early exit crash

            if dist.is_initialized():
                dist.barrier() # Commented out barrier as well
            time.sleep(3)
            exit(0)

    else:
        # --- ENGINE-ONLY PATH ---
        output_engine = self._forward_engine_attention(
            x,
            mask=mask,
            position_ids=position_ids,
            past_key_value_state=past_key_value_state,
            use_cache=use_cache,
            is_causal_mask=is_causal_mask,
            verbosity=debug_verbosity, # Pass verbosity level
        )

        if use_cache:
            x, cache, debug_engine = output_engine
        else:
            x, _, debug_engine = output_engine
            cache = None

        if debug_verbosity > 0 and debug_engine:
            for k, v in debug_engine.items(): # Keys already have prefix and rank suffix from _forward_engine_attention
                debug_info[k] = v # Directly assign

    return (x, cache) if use_cache else x



def _diff_debug_dicts(self, d1, d2, debug_verbosity=0): # Accept verbosity
    diffs = {}


    def normalize(key, prefix, is_engine=False):
        # Strip prefix
        base = key[len(prefix):]
        # Engine keys use _r{block_id}, Ring keys use _r{rank}
        # For comparison, we treat block_id as equivalent to rank
        # No need to collapse repeated suffixes as that was fixed.
        return base

    # Build reverse maps
    ring_map = {}
    for k in d1:
            if k.startswith("ring_"):
                ring_map[normalize(k, "ring_")] = k

    engine_map = {}
    for k in d2:
        if k.startswith("engine_"):
                engine_map[normalize(k, "engine_", is_engine=True)] = k

    shared = ring_map.keys() & engine_map.keys()
    missing_ring = sorted(set(engine_map.keys()) - set(ring_map.keys()))
    missing_engine = sorted(set(ring_map.keys()) - set(engine_map.keys()))

    # Print missing keys if verbosity >= 2
    if debug_verbosity >= 4:
        if missing_ring:
            print("Missing in Ring:")
            for k in missing_ring:
                print(f"  {engine_map[k]}")

        if missing_engine:
            print("Missing in Engine:")
            for k in missing_engine:
                print(f"  {ring_map[k]}")

    # --- Add Summary of Value Matches ---
    matching_keys = []
    non_matching_keys = []
    comparison_failed_keys = []

    for suffix in sorted(shared):
        rk, ek = ring_map[suffix], engine_map[suffix]
        val1, val2 = d1[rk], d2[ek]

        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            # Determine shorter tensor length for comparison
            n_compare = min(val1.numel(), val2.numel())
            if n_compare > 0:
                v1_flat = val1.flatten()[:n_compare]
                v2_flat = val2.flatten()[:n_compare]
                try:
                    # Use torch.allclose on the shorter length with relative tolerance (1%)
                    # Added atol for stability near zero
                    # Ensure tensors are on the same device (CPU) and same dtype for comparison
                    v1_flat = v1_flat.cpu().float()
                    v2_flat = v2_flat.cpu().float()
                    if torch.allclose(v1_flat.float(), v2_flat.float(), rtol=0.01, atol=1e-5):
                        matching_keys.append(suffix)
                    else:
                        non_matching_keys.append(suffix)
                except Exception as e:
                    comparison_failed_keys.append(f"{suffix} (Error: {e})")
            else:
                    # If tensors are empty or only one is, consider non-matching
                    non_matching_keys.append(suffix)
        else:
            # Non-tensor types or mismatched types are considered non-matching for this check
            non_matching_keys.append(suffix)

    # --- End Summary ---

    # Print the Value Match Summary if verbosity >= 1
    # (The detailed norm diffs below cover the minimal diff case)
    if debug_verbosity >= 1:
        summary_str = "\n--- Value Match Summary (First 5 Vals, 1% Tolerance) ---\n"
        summary_str += f"Matching Keys ({len(matching_keys)}): {sorted(matching_keys)}\n"
        summary_str += f"Non-Matching Keys ({len(non_matching_keys)}): {sorted(non_matching_keys)}\n"
        if comparison_failed_keys:
            summary_str += f"Comparison Failed Keys ({len(comparison_failed_keys)}): {sorted(comparison_failed_keys)}\n"
        summary_str += "--------------------------------------------------------\n"
        print(summary_str) # Print summary before detailed diffs

    # If verbosity >= 1, print norm diffs for non-matching keys
    if debug_verbosity >= 1:
        # Rename header for clarity
        print("--- Summary Diffs for Non-Matching Keys ---")
        for suffix in sorted(non_matching_keys):
            if suffix not in shared: continue # Skip if key wasn't actually shared (e.g., comparison failed)
            rk, ek = ring_map[suffix], engine_map[suffix]
            val1, val2 = d1[rk], d2[ek]
            if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                n_compare = min(val1.numel(), val2.numel())
                if n_compare > 0:
                    # Calculate norm diff over the shorter length
                    v1_flat = val1.flatten()[:n_compare].cpu().float()
                    v2_flat = val2.flatten()[:n_compare].cpu().float()
                    abs_diff = torch.abs(v1_flat - v2_flat)
                    norm_v2 = torch.linalg.norm(v2_flat).item()
                    norm_diff = torch.linalg.norm(v1_flat - v2_flat).item() # Calculate norm_diff first
                    rel_norm_diff = (norm_diff / norm_v2) if norm_v2 > 1e-9 else float('inf') # Avoid division by zero
                    max_diff = torch.max(abs_diff).item()
                    max_diff_flat_idx = torch.argmax(abs_diff).item()
                    mean_abs_diff = torch.mean(abs_diff).item()
                    threshold = 1e-4 # Use a threshold for significant difference
                    offending_indices = torch.where(abs_diff > threshold)[0]
                    num_offending = offending_indices.numel()
                    percentage_offending = (num_offending / abs_diff.numel()) * 100 if abs_diff.numel() > 0 else 0

                    # Unravel the flat index to multi-dimensional index
                    original_shape = val1.shape # Use val1's shape as reference
                    indices = []
                    temp_idx = max_diff_flat_idx
                    for dim_size in reversed(original_shape):
                        indices.append(temp_idx % dim_size)
                        temp_idx //= dim_size
                    max_diff_coords = tuple(reversed(indices))

                    print(f"  {suffix}: L2={norm_diff:.4f}, RelL2={rel_norm_diff:.4f}, MaxAbs={max_diff:.4f} @{max_diff_coords}, MeanAbs={mean_abs_diff:.4f}, Offending(>{threshold:.1e})={num_offending}/{n_compare} ({percentage_offending:.1f}%)")
                else:
                    # Handle case where one or both tensors are empty
                    print(f"  {suffix}: N/A (Empty Tensor)")
            else:
                print(f"  {suffix}: N/A (Non-Tensor or Type Mismatch)")
        print("--------------------------------------------------------")
    # If verbosity >= 3, print the full diffs
    if debug_verbosity >= 2: # Changed condition from >= 3 to >= 2
        # Indent this whole block
            print("\n--- Detailed Diffs ---") # Added header for clarity
            for suffix in sorted(shared): # Iterate through shared keys
                rk, ek = ring_map[suffix], engine_map[suffix]
                val1, val2 = d1[rk], d2[ek]

                diff_lines = [f"\n--- Key Suffix: {suffix} ---",
                            f"                    {'Shape':<25} {'Dtype':<15} {'First 5 Vals'}"] # Revert header

                if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                    # Print first 5 values
                    n_print = min(val1.numel(), val2.numel(), 5) # Limit to 5 values
                    v1_flat = val1.flatten()[:n_print].cpu().float()
                    v2_flat = val2.flatten()[:n_print].cpu().float()
                    v1 = [f"{v:.4f}" for v in v1_flat.tolist()]
                    v2 = [f"{v:.4f}" for v in v2_flat.tolist()]
                    diff_lines.append(f"Ring:   {str(val1.shape):<25} {str(val1.dtype):<15} {v1}")
                    diff_lines.append(f"Engine: {str(val2.shape):<25} {str(val2.dtype):<15} {v2}")

                    # Add detailed stats only for non-matching tensors in detailed view
                    if suffix in non_matching_keys:
                        n_compare = min(val1.numel(), val2.numel())
                        if n_compare > 0:
                            v1_flat_comp = val1.flatten()[:n_compare].cpu().float()
                            v2_flat_comp = val2.flatten()[:n_compare].cpu().float()
                            norm_diff = torch.linalg.norm(v1_flat_comp - v2_flat_comp).item()
                            abs_diff = torch.abs(v1_flat_comp - v2_flat_comp)
                            max_diff = torch.max(abs_diff).item()
                            max_diff_flat_idx = torch.argmax(abs_diff).item()
                            mean_abs_diff = torch.mean(abs_diff).item()
                            threshold = 1e-4 # Use the same threshold as summary
                            offending_indices = torch.where(abs_diff > threshold)[0]
                            num_offending = offending_indices.numel()
                            percentage_offending = (num_offending / abs_diff.numel()) * 100 if abs_diff.numel() > 0 else 0

                            # Unravel the flat index to multi-dimensional index
                            original_shape = val1.shape # Use val1's shape as reference
                            indices = []
                            temp_idx = max_diff_flat_idx
                            for dim_size in reversed(original_shape):
                                indices.append(temp_idx % dim_size)
                                temp_idx //= dim_size
                            max_diff_coords = tuple(reversed(indices))
                            # Find top 5 largest absolute differences
                            # Ensure k is not larger than the number of elements
                            top_k_diffs, top_k_indices = torch.topk(abs_diff, k=min(5, n_compare))
                            top_diff_details = []
                            for i in range(top_k_indices.numel()):
                                idx = top_k_indices[i].item()
                                top_diff_details.append(f"idx {idx}: Ring={v1_flat_comp[idx]:.4f}, Engine={v2_flat_comp[idx]:.4f} (Diff={top_k_diffs[i]:.4f})")
                            diff_lines.append(f"  Stats: L2 Norm Diff={norm_diff:.6f}, Max Abs Diff={max_diff:.6f} @{max_diff_coords}, Offending (> {threshold:.1e}): {num_offending}/{n_compare} ({percentage_offending:.2f}%)")
                            diff_lines.append(f"  Stats: Mean Abs Diff={mean_abs_diff:.6f}")
                            diff_lines.append(f"  Top 5 Abs Diffs: [{', '.join(top_diff_details)}]")

                elif isinstance(val1, tuple) and isinstance(val2, tuple):
                    # Basic tuple comparison (can be expanded if needed)
                    diff_lines.append(f"Ring:   {type(val1)} len={len(val1)}")
                    diff_lines.append(f"Engine: {type(val2)} len={len(val2)}")
                    if len(val1) != len(val2):
                        diff_lines.append("  Length mismatch")
                else:
                    diff_lines.append(f"  Mismatched types: Ring={type(val1)}, Engine={type(val2)}")

                diffs[suffix] = "\n".join(diff_lines) # Store the detailed diff string
                print(diffs[suffix]) # Print detailed diff immediately
    return diffs




def _gather_debug_tensors(self, data, strategy):
    """Recursively gathers tensors (if sharded) and returns CPU copy."""
    if isinstance(data, torch.Tensor):
        try:
            if isinstance(strategy, RingAttentionStrategy) and strategy.world_size > 1:
                if data.ndim == 3:
                    dim = 1
                elif data.ndim == 4:
                    dim = 2
                else:
                    return data.detach().cpu()
                return strategy.gather_tensor(data, dim=dim).detach().cpu()
        except Exception as e:
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(f"[Rank {rank}] Warning: gather failed: {e}")
        return data.detach().cpu()

    elif isinstance(data, dict):
        return {k: self._gather_debug_tensors(v, strategy) for k, v in data.items()}

    elif isinstance(data, (list, tuple)):
        return type(data)(self._gather_debug_tensors(v, strategy) for v in data)

    return data  # Scalar or unknown type


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
    verbosity: int, # Add verbosity
    rank=0,
):
    debug = {}
    enable_debug_info = verbosity > 0 # Use verbosity to determine if debug is enabled
    
    if enable_debug_info:
        # Log the local input x before gathering
        debug[f"ring_x_local_input_r{rank}"] = x.clone().detach().cpu()
    
    residual = x
    
    # Apply LayerNorm locally
    x_norm_local = self.ln(x) 
    if enable_debug_info:
        # Log the local norm. The comparison function expects keys ending in _block_r{rank}
        # Note: This tensor might have padding applied later in the helper for FF.
        # The helper should log the *unpadded* version for accurate comparison.
        debug[f"ring_x_norm_local_pre_pad_r{rank}"] = x_norm_local.clone().detach().cpu() 

    ring_helper = RingAttentionHelper(
        attn_module=self.attn,
        strategy=strategy,
        llama_block=self,
        use_cache=use_cache,
        debug_mode=(verbosity >= 1), # Enable internal debug only at level 4
        ff=self.ff_sub_layer,              # Match engine
        ff_norm=self.ff_ln,                # Match engine
    )

    local_x_shape = x.shape
    correct_valid_len = strategy._local_valid_len # Use the strategy's valid length
    # print(f"[Rank {rank}, Layer {self.layer_index}] RingAttention: Input x shape={local_x_shape}, Correct valid_len={correct_valid_len}") # DEBUG PRINT

    x, cache, debug_ring = ring_helper.forward(
        x_norm_local, # Pass the local normalized tensor
        mask=mask,
        strategy=strategy,
        position_ids=position_ids,
        past_key_value_state=past_key_value_state,
        is_causal_mask=is_causal_mask,
        rank=rank,
        valid_len=correct_valid_len, # Pass the correct valid length
        residual=residual # Pass the local residual source
    )


    if enable_debug_info and debug_ring:
        for k, v in debug_ring.items():
            debug[f"ring_{k}"] = v

    # Always return the locally created debug dict if enable_debug_info is true
    # Return None for debug dict if verbosity is 0
    if verbosity == 0:
        return x, cache, None
    else:
        return x, cache, debug

def _forward_engine_attention(
    self,
    x,
    *,
    mask,
    position_ids,
    past_key_value_state,
    use_cache,
    is_causal_mask,
    verbosity: int, # Add verbosity
):
    debug = {}
    enable_debug_info = verbosity > 0 # Use verbosity to determine if debug is enabled
    # Log the input x received by this function
    if enable_debug_info:
        debug[f"engine_x_input_r{dist.get_rank() if dist.is_initialized() else 0}"] = x.clone().detach().cpu() # Add engine_ prefix
    x_norm = self.ln(x) # This is the global x_norm
    if enable_debug_info:
        debug[f"engine_x_norm_r{dist.get_rank() if dist.is_initialized() else 0}"] = x_norm.clone().detach().cpu() # Add engine_ prefix

    queries, keys, values = self.compute_qkv_and_rope_thread(
        self.attn,
        q=x_norm,
        k=x_norm,
        v=x_norm,
        position_ids=position_ids,
        use_cache=use_cache,
        past_key_value_state=past_key_value_state,
        is_self=True,
    )

    if use_cache and past_key_value_state and past_key_value_state[0].numel() > 0:
        keys = torch.cat((past_key_value_state[0], keys), dim=2)
        values = torch.cat((past_key_value_state[1], values), dim=2)

    expansion = self.attn.nheads // self.attn.kvheads
    keys_e = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2) if expansion != 1 else keys
    values_e = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2) if expansion != 1 else values

    engine = ThreadedRingAttentionEngine(
        block_size=32,
        attn=self.attn,
        ff=self.ff_sub_layer,
        ff_norm=self.ff_ln,
        is_causal=is_causal_mask and mask is None,
        debug_mode=(verbosity >= 1), # Enable internal debug only at level 4
    )


    engine_output = engine.forward_full(
        q_global=queries,
        k_global=keys_e,
        v_global=values_e,
        mask_global=mask,
        x_global=x,
        x_norm_global=x_norm, # Pass global x_norm
    )


    if enable_debug_info:
        if verbosity >= 1: # Corrected: Only unpack debug data if engine's debug mode was enabled
            x, engine_debug_data = engine_output
        else:
            x = engine_output # Otherwise, only the output tensor is returned
            engine_debug_data = None # Set to None as it wasn't returned

        if engine_debug_data: # Check if debug data exists before processing
            # Add engine prefix to keys coming from the engine's debug buffer
            for k, v in engine_debug_data.items():
                debug[f"engine_{k}"] = v

    else:
        x = engine_output


    cache = (keys, values) if use_cache else None
    return x, cache, debug