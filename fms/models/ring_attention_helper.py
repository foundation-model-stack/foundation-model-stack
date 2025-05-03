from typing import List, Tuple, Dict, Optional, Union
import torch
import torch.distributed as dist
import math

def _pad_to_block(t, target_len, dim=2):
    """Pads a tensor `t` to `target_len` along dimension `dim` with zeros."""
    pad_len = target_len - t.shape[dim]
    if pad_len <= 0:
        return t
    pad_shape = list(t.shape)
    pad_shape[dim] = pad_len
    pad = torch.zeros(*pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)

class RingAttentionHelper:
    def __init__(self, attn_module, strategy, llama_block,use_cache=False,
             debug_mode: bool = False, minimal_debug_prints: bool = False,
             ff=None, ff_norm=None):  # <-- Add these
        self.attn = attn_module
        self.ff = ff
        self.ff_norm = ff_norm
        self.attn = attn_module
        self.strategy = strategy # Assuming strategy contains block_size
        self.use_cache = use_cache
        self.debug_mode = debug_mode
        self.llama_block = llama_block
        self.minimal_debug_prints = minimal_debug_prints
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.head_dim = attn_module.emb_kq_per_head
        self.scale = math.sqrt(self.head_dim)
        # Ensure block_size is set
        if not hasattr(self.strategy, 'block_size'):
             print("Warning: strategy object does not have 'block_size'. Using a default of 128.")
             self.strategy.block_size = 128

    def forward(self, x_norm, strategy, mask=None, position_ids=None, past_key_value_state=None,
            is_causal_mask=False, rank=0, minimal_debug_prints: bool = False, valid_len=0, 
            residual=None):
        """Main forward pass, delegates to forward_full after initial setup without global gather."""
        
        start_idx_global = self.rank * self.strategy.block_size
        B, T = x_norm.shape[:2]

        if position_ids is None:
            # Initialize to -1 by default
            position_ids = torch.full((B, T), fill_value=-1, dtype=torch.long, device=x_norm.device)
            
            if valid_len > 0:
                valid_pos = torch.arange(start_idx_global, start_idx_global + valid_len, device=x_norm.device)
                position_ids[:, :valid_len] = valid_pos.unsqueeze(0)  # Broadcast to all batches


        # Compute local QKV aligned to global rotary positions
        q_local, k_local, v_local = self.llama_block.compute_local_qkv_and_rope(
            self.attn,
            q=x_norm, k=x_norm, v=x_norm,
            position_ids=position_ids,
            use_cache=False,
            past_key_value_state=past_key_value_state,
            is_self=True
        )

        # Trim QKV tensors to valid_len (not block_size)
        q_local = q_local[:, :, :valid_len, :]
        k_local = k_local[:, :, :valid_len, :]
        v_local = v_local[:, :, :valid_len, :]

        # Ensure x_norm and residual are trimmed to valid_len if they weren't already
        # (They should be based on how _forward_ring_attention calls this)
        x_norm_local = x_norm[:, :valid_len, :]
        residual_local = residual[:, :valid_len, :] if residual is not None else None

        # Forward full with locally computed Q/K/V
        result = self.forward_full(
            q_local=q_local,
            k_local=k_local,
            v_local=v_local,
            mask_global=mask,
            x_block=residual_local, # Pass trimmed residual
            x_norm_block=x_norm_local, # Pass trimmed norm
            valid_len=valid_len,
            q_start_global=start_idx_global
        )

        if self.debug_mode:
            x, debug_info = result
            return x, None, debug_info
        else:
            return result, None, None


    def forward_full(self, q_local: torch.Tensor, k_local: torch.Tensor, v_local: torch.Tensor,
                     mask_global: Optional[torch.Tensor], # x_global: torch.Tensor, x_norm_global: torch.Tensor,
                     valid_len: int, x_block, x_norm_block, q_start_global) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict],]:
        """
        Performs the full ring attention forward pass using a two-pass approach.
        Uses torch.distributed for communication.
        """
        debug_info = {} if self.debug_mode else None

        B, H, T_q_local, D_head = q_local.shape
        D_v = self.attn.emb_v_per_head
        T_q_local = valid_len # Use the provided valid_len as the local sequence length
        T_block = self.strategy.block_size # Padded block size per rank

        # Determine the start and end indices for the local block on this rank
        start_idx_global = self.rank * self.strategy.block_size
        # Use T_q_local (valid_len) for slicing Q, K, V from the global QKV tensors
        end_idx_qkv = start_idx_global + T_q_local
        # Use T_block for slicing x_global and x_norm_global to match engine's block slicing
        # end_idx_block = start_idx_global + T_block # Not needed if we receive trimmed inputs

        if self.debug_mode and debug_info is not None:
            debug_info.update({
                f"q_local_r{self.rank}": q_local.detach().cpu(), # Shape: [B, H, valid_len, D]
                f"k_local_r{self.rank}": k_local.detach().cpu(), # Shape: [B, Hkv, valid_len, D]
                # Log the trimmed x_block (residual) and x_norm_block received
                f"v_local_r{self.rank}": v_local.detach().cpu(), # Shape: [B, Hkv, valid_len, Dv]
                f"x_norm_r{self.rank}": x_norm_block.detach().cpu(), # Shape: [B, valid_len, D_emb]
                f"x_block_r{self.rank}": x_block.detach().cpu(),
            })

        # --- Pass 1: Compute Max Scores ---
        max_score = self._compute_max_score_pass(
            q_local=q_local,
            k_local=k_local,
            mask_global=mask_global,
            q_start_global=start_idx_global,
            valid_len_local=T_q_local,
            debug_info=debug_info
        )

        if self.debug_mode and debug_info is not None:
             debug_info[f"max_score_r{self.rank}"] = max_score.clone().detach().cpu()


        # --- Pass 2: Compute Numerator and Denominator ---
        numerator, denominator = self._compute_sums_pass(
            q_local=q_local,
            k_local=k_local,
            v_local=v_local,
            mask_global=mask_global,
            q_start_global=start_idx_global,
            valid_len_local=T_q_local,
            max_score=max_score,
            debug_info=debug_info
        )

        if self.debug_mode and debug_info is not None:
            debug_info.update({
                f"numerator_r{self.rank}": numerator.clone().detach().cpu(),
                f"denominator_r{self.rank}": denominator.clone().detach().cpu()
            })

        # --- Final Attention Output Calculation ---
        attn_out_h = numerator / (denominator + 1e-10)
        attn_out = attn_out_h.to(q_local.dtype)
        B, H, T_q_local, D_v = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q_local, H * D_v)

        # Apply dense layer and dropout
        attn_out = self.attn.dense(attn_out)
        # if hasattr(self.llama_block, 'dropout') and self.llama_block.config.p_dropout != 0:
        #     attn_out = self.llama_block.dropout(attn_out)

        if self.debug_mode and debug_info is not None:
            debug_info[f"attn_out_raw_r{self.rank}"] = attn_out.clone().detach().cpu()

        # Add residual connection for attention output (using trimmed tensors)
        residual_1 = x_block + attn_out
        if self.debug_mode and debug_info is not None:
            debug_info[f"attn_out_residual_r{self.rank}"] = residual_1.clone().detach().cpu()

        # --- Feedforward Network ---
        # Pad residual_1 before FF Norm and FF layer
        residual_1_padded = _pad_to_block(residual_1, self.strategy.block_size, dim=1)

        ff_ln_out_padded = self.ff_norm(residual_1_padded)
        if self.debug_mode and debug_info is not None:
            # Log the padded version going into FF
            debug_info[f"ff_ln_out_padded_r{self.rank}"] = ff_ln_out_padded.clone().detach().cpu()

        ff_out_padded = self.ff(ff_ln_out_padded)
        if self.debug_mode and debug_info is not None:
            debug_info[f"ff_out_raw_padded_r{self.rank}"] = ff_out_padded.clone().detach().cpu()

        # if hasattr(self.llama_block, 'dropout') and self.llama_block.config.p_dropout != 0:
        #     ff_out_raw = self.llama_block.dropout(ff_out_raw)

        # Slice FF output back to valid_len before adding residual
        ff_out_trimmed = ff_out_padded[:, :T_q_local, :]
        x = ff_out_trimmed + residual_1 # Add trimmed tensors

        if self.debug_mode and debug_info is not None:
            debug_info[f"block_output_r{self.rank}"] = x.clone().detach().cpu()

        return (x, debug_info) if self.debug_mode else x

    def _compute_max_score_pass(self, q_local: torch.Tensor, k_local: torch.Tensor,
                                mask_global: Optional[torch.Tensor], q_start_global: int,
                                valid_len_local: int, debug_info: Optional[Dict]) -> torch.Tensor:
        """
        First pass: Computes maximum attention scores for numerical stability.
        """
        B, H, T_q_local, D_head = q_local.shape
        device = q_local.device
        dtype = torch.float32 # Use float32 for scores

        max_score = torch.full((B, H, T_q_local, 1), -float("inf"), device=device, dtype=dtype)

        # Global indices for the local query block
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)

        current_k_block = k_local.clone() # Start with the local k block
        current_k_len = k_local.shape[2] # Valid length of the current k block

        for i in range(self.world_size):
            # Global start index of the current k block being processed
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.strategy.block_size
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)


            # Get the relevant slice of the global mask
            # Note: mask slicing uses global indices
            current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len] if mask_global is not None else None
            if current_k_len == 0:
                # Need to shift tensors even if we skip computation
                if i < self.world_size - 1:
                    current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                continue 
            # Compute raw scores for debugging before applying masks
            if self.debug_mode and debug_info is not None:
                 raw_scores = self._compute_attention_scores(
                    q=q_local, k=current_k_block[:, :, :current_k_len, :], # Use current_k_len for slicing
                    q_indices_global=q_indices_global, k_indices_global=k_indices_global,
                    mask=None, # No mask for raw scores
                    apply_mask=False, keep_causal=False # Do not apply any mask
                )
                 debug_info[f"raw_scores_step{i}_r{self.rank}"] = raw_scores.clone().detach().cpu()


            # Compute attention scores with masks
            scores = self._compute_attention_scores(
                q=q_local, k=current_k_block[:, :, :current_k_len, :], # Use current_k_len for slicing
                q_indices_global=q_indices_global, k_indices_global=k_indices_global,
                mask=current_mask,
                apply_mask=True # Apply both causal and padding masks
            )

            if self.debug_mode and debug_info is not None:
                debug_info[f"scores_step{i}_r{self.rank}"] = scores.clone().detach().cpu()

            # Update the maximum score
            max_score = self._update_max_score(scores, max_score)

            # Ring shift k block (if not the last iteration)
            if i < self.world_size - 1:
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)

                if self.debug_mode and debug_info is not None:
                    debug_info[f"k_input_step{i+1}_r{self.rank}"] = current_k_block[:, :, :current_k_len, :].clone().detach().cpu()


        return max_score


    def _compute_sums_pass(self, q_local: torch.Tensor, k_local: torch.Tensor, v_local: torch.Tensor,
                           mask_global: Optional[torch.Tensor], q_start_global: int,
                           valid_len_local: int, max_score: torch.Tensor, debug_info: Optional[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Second pass: Computes numerator and denominator for the softmax.
        """
        B, H, T_q_local, D_head = q_local.shape
        D_v = self.attn.emb_v_per_head
        device = q_local.device
        dtype = torch.float32 # Use float32 for numerator and denominator

        numerator = torch.zeros(B, H, T_q_local, D_v, device=device, dtype=dtype)
        denominator = torch.zeros(B, H, T_q_local, 1, device=device, dtype=dtype)

        # Global indices for the local query block
        q_indices_global = torch.arange(q_start_global, q_start_global + T_q_local, device=device)

        current_k_block = k_local.clone() # Start with the local k block
        current_v_block = v_local.clone() # Start with the local v block
        current_k_len = k_local.shape[2] # Valid length of the current k block

        for i in range(self.world_size):
            # Global start index of the current k block being processed
            k_start_global = ((self.rank - i + self.world_size) % self.world_size) * self.strategy.block_size
            k_indices_global = torch.arange(k_start_global, k_start_global + current_k_len, device=device)

            # Skip computation if the received K/V block has zero length
            if current_k_len == 0:
                # Need to shift tensors even if we skip computation
                if i < self.world_size - 1:
                    current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                    current_v_block, _ = self._ring_shift_tensor(current_v_block, self.strategy.block_size)
                continue # Skip to the next iteration

            # Get the relevant slice of the global mask
            current_mask = mask_global[:, :, q_start_global:q_start_global+T_q_local, k_start_global:k_start_global+current_k_len] if mask_global is not None else None

            # Compute attention scores with masks
            scores = self._compute_attention_scores(
                q=q_local, k=current_k_block[:, :, :current_k_len, :], # Use current_k_len for slicing
                q_indices_global=q_indices_global, k_indices_global=k_indices_global,
                mask=current_mask,
                apply_mask=True
            )

            # Update numerator and denominator
            numerator, denominator = self._update_totals(scores, current_v_block[:, :, :current_k_len, :], max_score, numerator, denominator) # Use current_k_len for slicing v

            # Ring shift k and v blocks (if not the last iteration)
            if i < self.world_size - 1:
                current_k_block, current_k_len = self._ring_shift_tensor(current_k_block, self.strategy.block_size)
                current_v_block, _ = self._ring_shift_tensor(current_v_block, self.strategy.block_size) # Assuming v has same length as k

                if self.debug_mode and debug_info is not None:
                     debug_info[f"k_input_step{i+1}_r{self.rank}"] = current_k_block[:, :, :current_k_len, :].clone().detach().cpu()
                     debug_info[f"v_input_step{i+1}_r{self.rank}"] = current_v_block[:, :, :current_k_len, :].clone().detach().cpu()


        return numerator, denominator

    def _compute_attention_scores(self, q: torch.Tensor, k: torch.Tensor, q_indices_global: torch.Tensor,
                                 k_indices_global: torch.Tensor, mask: Optional[torch.Tensor] = None,
                                 apply_mask: bool = True, keep_causal: bool = True) -> torch.Tensor:
        """Computes attention scores, optionally applying padding and causal masks."""
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / self.scale

        if apply_mask and mask is not None:
            scores = scores + mask.to(scores.dtype)

        if apply_mask and keep_causal:
             causal_mask = (k_indices_global[None, :] > q_indices_global[:, None]).unsqueeze(0).unsqueeze(0)
             scores = scores.masked_fill(causal_mask, -torch.inf)

        return scores

    def _update_max_score(self, scores: torch.Tensor, current_max: torch.Tensor) -> torch.Tensor:
        """Updates max score handling -inf."""
        block_max = scores.masked_fill(scores == -torch.inf, torch.finfo(scores.dtype).min).amax(dim=-1, keepdim=True)
        return torch.maximum(current_max, block_max)

    def _update_totals(self, scores: torch.Tensor, v: torch.Tensor, max_score: torch.Tensor,
                      numerator: torch.Tensor, denominator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates numerator and denominator using stable exponentiation."""
        stable_scores = (scores - max_score).clamp(min=-10.0, max=10.0)
        exp_scores = torch.exp(stable_scores)

        exp_scores = torch.nan_to_num(exp_scores, nan=0.0, posinf=torch.finfo(exp_scores.dtype).max, neginf=0.0)

        numerator += torch.einsum("bhqk,bhkd->bhqd", exp_scores, v.to(numerator.dtype))

        denominator += exp_scores.sum(dim=-1, keepdim=True)

        return numerator, denominator


    def _ring_shift_tensor(self, tensor: torch.Tensor, pad_len: int) -> Tuple[torch.Tensor, int]:
        """
        Ring‑shifts `tensor` along its last dimension by one rank:
        - CPU: non‑blocking isend/irecv.
        - GPU: all_gather of lengths + all_to_all of only the neighbor’s block.
        Returns (received_tensor cropped to true length, received_length).
        """
        rank, world = self.rank, self.world_size
        send_rank = (rank + 1) % world
        recv_rank = (rank - 1 + world) % world

        # 1) pad along last dim
        valid_len = tensor.shape[-1]
        padded = _pad_to_block(tensor, pad_len, dim=-1).contiguous()

        if not tensor.is_cuda:
            """
            Performs a ring shift of a tensor using torch.distributed.
            Pads to pad_len, sends, receives, and returns received tensor and its valid length.
            """
            send_rank = (self.rank + 1) % self.world_size
            recv_rank = (self.rank - 1 + self.world_size) % self.world_size

            valid_len = tensor.shape[2]
            padded = _pad_to_block(tensor, pad_len, dim=2).contiguous()

            send_len = torch.tensor([valid_len], dtype=torch.int32, device=tensor.device)
            recv_len = torch.empty(1, dtype=torch.int32, device=tensor.device)

            reqs = [
                dist.isend(send_len, dst=send_rank),
                dist.irecv(recv_len, src=recv_rank)
            ]
            for req in reqs:
                req.wait()

            tensor_recv = torch.empty_like(padded)
            reqs = [
                dist.isend(padded, dst=send_rank),
                dist.irecv(tensor_recv, src=recv_rank)
            ]
            for req in reqs:
                req.wait()

            # print(f"[Rank {rank}][CPU] tensor_recv.shape={tuple(tensor_recv.shape)}, recv_len={recv_len}")

            return tensor_recv, recv_len.item()

        else:
            # GPU: pure‑collective version that mirrors the CPU branch exactly

            # 1) determine how many blocks we really have, then pad that block axis
            #    tensor is [..., blocks, block_size]
            valid_len = tensor.size(-2)  
            padded    = _pad_to_block(tensor, pad_len, dim=-2).contiguous()
            # now padded.shape[-2] == pad_len, padded.shape[-1] == block_size

            # 2) exchange block‑counts via all_gather
            len_t    = torch.tensor([valid_len], dtype=torch.int32, device=tensor.device)
            len_list = [torch.empty_like(len_t) for _ in range(world)]
            dist.all_gather(len_list, len_t)
            recv_len = int(len_list[recv_rank].item())

            # 3) build uniform‐shape send/recv lists
            send_list = [
                padded.clone() if r == send_rank else torch.zeros_like(padded)
                for r in range(world)
            ]
            recv_list = [torch.empty_like(padded) for _ in range(world)]

            # 4) do the all_to_all
            dist.all_to_all(recv_list, send_list)

            # 5) pick exactly the CPU‐style return values
            tensor_recv = recv_list[recv_rank]

            # 6) debug print matches CPU’s: shape of the full padded buffer + the true block count
            # print(f"[Rank {rank}][GPU] tensor_recv.shape={tuple(tensor_recv.shape)}, recv_len={recv_len}")

            # 7) return (full padded buffer, true block count)
            return tensor_recv, recv_len
