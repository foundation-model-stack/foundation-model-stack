import math
from typing import List, Optional, Tuple

from fms.modules.attention import (
    AttentionKwargs,
    _sdpa_compute_op,
    register_attention_op,
)
import torch

from torch.library import custom_op
import torch.nn.functional as F


@custom_op("spyre::paged_attn_store", mutates_args=(), device_types=["cpu", "cuda"])
def paged_attn_store(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    result_key_cache = key_cache.clone()
    result_value_cache = value_cache.clone()
    block_size = value_cache.shape[1]
    for seq_i, slot_mapping_seq in enumerate(slot_mapping):
        for tok_i, slot in enumerate(slot_mapping_seq):
            block_number = slot.item() // block_size
            position = slot.item() % block_size

            result_key_cache[block_number, position, :, :] = key[seq_i, tok_i, :, :]
            result_value_cache[block_number, position, :, :] = value[seq_i, tok_i, :, :]
    return result_key_cache, result_value_cache


@paged_attn_store.register_fake
def paged_attn_store_meta(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return key_cache, value_cache


@custom_op("spyre::paged_attn_compute", mutates_args={}, device_types=["cpu", "cuda"])
def paged_attn_compute(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    current_tkv_mask: torch.Tensor,
    left_padded_prompt_mask: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    # torch.zeros(NUM_BLOCKS, BLOCK_SIZE, kvheads, head_size, dtype=model_dtype),
    output = torch.zeros_like(query)
    num_query_heads = query.shape[2]
    num_kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
    block_size = value_cache.shape[1]
    seq_len_q = query.shape[1]
    num_seqs = query.shape[0]

    block_tables_lst = block_table.tolist()

    seq_lens_lst = current_tkv_mask.tolist()
    for i in range(num_seqs):
        q = query[i]
        block_table = block_tables_lst[i]
        start_pos = int(left_padded_prompt_mask[i].item())
        seq_len = int(seq_lens_lst[i])
        seq_len_q_i = seq_len_q

        keys_lst: list[torch.Tensor] = []
        values_lst: list[torch.Tensor] = []
        for j in range(start_pos, seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, block_offset, :, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, block_offset, :, :]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        seq_len_kv = keys.shape[0]

        # cut the pads for first prefill
        if q.shape[0] > seq_len_kv:
            seq_len_q_i = seq_len_kv
            q = q[-seq_len_kv:]

        if num_kv_heads > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_query_heads // num_kv_heads, dim=1)
            values = torch.repeat_interleave(
                values, num_query_heads // num_kv_heads, dim=1
            )

        # Generate mask for prefix attention
        mask = torch.ones((1, 1, seq_len_q_i, seq_len_kv), dtype=torch.bool)
        mask[:, :, :, -seq_len_q_i:] = torch.tril(mask[:, :, :, -seq_len_q_i:])
        mask = torch.where(mask.logical_not(), -torch.inf, 0.0).to(
            device=query.device, dtype=query.dtype
        )

        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),  # format for sdpa
            keys.transpose(0, 1).unsqueeze(0),  # format for sdpa
            values.transpose(0, 1).unsqueeze(0),  # format for sdpa
            attn_mask=mask,
            scale=scale,
        )

        out = out.transpose(1, 2).view(seq_len_q_i, num_query_heads, head_size)
        output[i][-seq_len_q_i:] = out
    return output


@paged_attn_compute.register_fake
def paged_attn_compute_meta(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    current_tkv_mask: torch.Tensor,
    left_padded_prompt_mask: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros_like(query)


class SpyrePagedAttentionKwargs(AttentionKwargs):
    current_tkv_mask: Optional[torch.Tensor]
    left_padded_prompt_mask: Optional[torch.Tensor]
    block_table: Optional[torch.Tensor]
    slot_mapping: torch.Tensor
    mask: Optional[torch.Tensor]  # prefill mask


def __spyre_paged_store_op(
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: Optional[torch.Tensor],
    value_cache: Optional[torch.Tensor],
    **attn_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    result_key_cache, result_value_cache = torch.ops.spyre.paged_attn_store(
        keys, values, key_cache, value_cache, attn_kwargs["slot_mapping"]
    )

    # for prefill, we want to return the original keys/values
    if attn_kwargs.get("block_table", None) is None:
        return keys, values, result_key_cache, result_value_cache
    else:
        return (
            result_key_cache,
            result_value_cache,
            result_key_cache,
            result_value_cache,
        )


def __spyre_paged_compute_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: Optional[float],
    **attn_kwargs,
) -> torch.Tensor:
    if scale_factor is None:
        scale_factor = 1 / math.sqrt(query.shape[-1])
    return torch.ops.spyre.paged_attn_compute(
        query,
        key_cache,
        value_cache,
        scale_factor,
        attn_kwargs["current_tkv_mask"],
        attn_kwargs["left_padded_prompt_mask"],
        attn_kwargs["block_table"],
    )


def __spyre_paged_validate_attn_kwargs_op(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_value_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    **attn_kwargs,
):
    """
    Validating the SpyrePagedAttentionKwargs for proper shapes as this will reduce the number of symbolic shapes to the minimum needed set
    """
    assert input_ids.shape[0] == position_ids.shape[0]
    assert input_ids.shape[1] == position_ids.shape[1]

    assert input_ids.shape[0] == attn_kwargs["slot_mapping"].shape[0]
    assert input_ids.shape[1] == attn_kwargs["slot_mapping"].shape[1]

    block_table = attn_kwargs.get("block_table", None)
    if block_table is not None:
        assert input_ids.shape[0] == block_table.shape[0]
    current_tkv_mask = attn_kwargs.get("current_tkv_mask", None)
    if current_tkv_mask is not None:
        assert input_ids.shape[0] == current_tkv_mask.shape[0]
    left_padded_prompt_mask = attn_kwargs.get("left_padded_prompt_mask", None)
    if left_padded_prompt_mask is not None:
        assert input_ids.shape[0] == left_padded_prompt_mask.shape[0]

    if past_key_value_states is not None:
        for k, v in past_key_value_states:
            # assert that for each layer, k and v have the same number of blocks
            assert k.shape[0] == v.shape[0]

            # assert that for a given layer, it has the same number of blocks as any other layer
            for i in range(len(past_key_value_states)):
                assert k.shape[0] == past_key_value_states[i][0].shape[0]
                assert v.shape[0] == past_key_value_states[i][1].shape[0]


register_attention_op(
    "spyre_paged_attn",
    __spyre_paged_store_op,
    _sdpa_compute_op,
    is_prefill_op=lambda **attn_kwargs: attn_kwargs.get("block_table", None) is None,
    compute_decode_op=__spyre_paged_compute_op,
    validate_attn_kwargs_op=__spyre_paged_validate_attn_kwargs_op,
)
