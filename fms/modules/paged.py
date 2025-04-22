from typing import Optional, Tuple

from fms.modules.attention import AttentionOp
import torch

from torch.library import custom_op

@custom_op("aiu::paged_attn_store", mutates_args=(), device_types="cpu")
def paged_attn_store(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    result_key_cache = key_cache.clone()
    result_value_cache = value_cache.clone()
    for seq_i, slot_mapping_seq in enumerate(slot_mapping):
        for tok_i, slot in enumerate(slot_mapping_seq):
            block_number = slot.item() // 64
            position = slot.item() % 64

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


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


@custom_op("aiu::paged_attn_compute", mutates_args={}, device_types="cpu")
def paged_attn_compute(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    partial_page_tkv_mask: torch.Tensor,
    left_padded_prompt_mask: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    output = torch.zeros_like(query)
    num_query_heads = query.shape[2]
    num_kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
    block_size = value_cache.shape[1]
    num_seqs = query.shape[0]

    block_tables_lst = block_table.cpu().tolist()
    # adding as all sizes will be same
    seq_lens_lst = (left_padded_prompt_mask + partial_page_tkv_mask).cpu().tolist()
    for i in range(num_seqs):
        q = query[i]
        block_table = block_tables_lst[i]
        start_pos = left_padded_prompt_mask[i].item()
        seq_len = int(seq_lens_lst[i])

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
        if num_kv_heads > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_query_heads // num_kv_heads, dim=1)
            values = torch.repeat_interleave(
                values, num_query_heads // num_kv_heads, dim=1
            )

        out = ref_masked_attention(q, keys, values, scale)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    return output


@paged_attn_compute.register_fake
def paged_attn_compute_meta(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    partial_page_tkv_mask: torch.Tensor,
    left_padded_prompt_mask: torch.Tensor,
    block_table: torch.Tensor,
) -> torch.Tensor:
    return torch.zeros_like(query)

class PagedAttentionOp(AttentionOp):

    def __init__(
        self, 
        slot_mapping: torch.Tensor, 
        block_table: Optional[torch.Tensor] = None, 
        partial_page_tkv_mask: Optional[torch.Tensor] = None, 
        left_padded_prompt_mask: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self._slot_mapping = slot_mapping
        self._scale = scale
        self._block_table = block_table
        self._partial_page_tkv_mask = partial_page_tkv_mask
        self._left_padded_prompt_mask = left_padded_prompt_mask

    @classmethod
    def from_decode_meta(
        cls, 
        slot_mapping: torch.Tensor,  
        block_table: torch.Tensor, 
        partial_page_tkv_mask: torch.Tensor, 
        left_padded_prompt_mask: torch.Tensor,
        scale: float,
    ):
        return cls(slot_mapping, block_table, partial_page_tkv_mask, left_padded_prompt_mask, scale)

    @classmethod
    def from_prefill_meta(cls, slot_mapping: torch.Tensor):
        return cls(slot_mapping)
    
    def is_prefill(self) -> bool:
        return self._block_table is None
    
    def store(self, keys, values, key_cache, value_cache):
        return torch.ops.aiu.paged_attn_store(
            keys, values, key_cache, value_cache, self._slot_mapping
        )
    
    def compute(self, query, key_cache, value_cache):
        return torch.ops.aiu.paged_attn_compute(
            query,
            key_cache,
            value_cache,
            self._scale,
            self._partial_page_tkv_mask,
            self._left_padded_prompt_mask,
            self._block_table,
        )
    
def prepare_model_inputs_hook(model: torch.nn.Module, num_blocks: int = 100, block_size: int = 64):
    if hasattr(model, "head"):
        model_dtype = model.head.weight.dtype
    elif hasattr(model, "shared"):
        model_dtype = model.shared.head.weight.dtype
    else:
        model_dtype = torch.float32
    
    nheads = model.config.nheads
    if hasattr(model.config, "kvheads"):
        kvheads = model.config.kvheads
    elif hasattr(model.config, "multiquery_attn"):
        kvheads = 1 if model.config.multiquery_attn else model.config.nheads
    else:
        kvheads = nheads

    tensor_parallel_size = (
        model.distributed_strategy.group.size()
        if hasattr(model.distributed_strategy, "group")
        else 1
    )
    kvheads = kvheads // tensor_parallel_size if kvheads > 1 else kvheads
    head_size = model.config.emb_dim // nheads

    block_numbers = [i for i in range(num_blocks)]
    global slot_mapping
    global block_table
    global left_padded_prompt_mask
    global partial_page_tkv_mask
    global position_i

    
    def _inner(generation_iter: int, input_ids: torch.Tensor, kwargs):
        global slot_mapping
        global block_table
        global left_padded_prompt_mask
        global partial_page_tkv_mask
        global position_i

        # prefill
        if generation_iter == 0:
            position_i = input_ids.size(1) - 1
            kwargs["past_key_value_states"] = [
                (
                    torch.zeros(num_blocks, block_size, kvheads, head_size, dtype=model_dtype),
                    torch.zeros(num_blocks, block_size, kvheads, head_size, dtype=model_dtype),
                )
                for _ in range(model.config.nlayers)
            ]
            left_padded_prompt_mask = (kwargs["position_ids"] == 0).sum(dim=1) - 1
            partial_page_tkv_mask = (kwargs["position_ids"] != 0).sum(dim=1) + 1

            slot_mapping = []
            block_table = []
            for seq_i in input_ids:
                block_table_i = []
                slot_mapping_i = []
                for pos_i in range(seq_i.size(0)):
                    if pos_i % block_size == 0:
                        block_number = block_numbers.pop(0)
                        block_table_i.append(block_number)
                    block_offset = pos_i % block_size
                    slot = block_number * block_size + block_offset
                    slot_mapping_i.append(slot)
                slot_mapping.append(slot_mapping_i)
                block_table.append(block_table_i)
            slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64)

            # kwargs["mask"] = kwargs["mask"].unsqueeze(1)

            # batch dynamic
            torch._dynamo.mark_static(input_ids, 0)
            torch._dynamo.mark_static(slot_mapping_tensor, 0)
            torch._dynamo.mark_static(kwargs["position_ids"], 0)
            torch._dynamo.mark_static(kwargs["mask"], 0)

            # seq dynamic
            torch._dynamo.mark_dynamic(slot_mapping_tensor, 1)
            torch._dynamo.mark_dynamic(kwargs["position_ids"], 1)
            torch._dynamo.mark_dynamic(kwargs["mask"], 2)
            torch._dynamo.mark_dynamic(kwargs["mask"], 3)
            kwargs["custom_attention_op"] = PagedAttentionOp.from_prefill_meta(
                slot_mapping_tensor
            )

        # decode
        else:
            kwargs["mask"] = None
            position_i = position_i + 1
            if position_i % block_size == 0:
                for block_table_i in block_table:
                    block_number = block_numbers.pop(0)
                    block_table_i.append(block_number)
            block_offset = position_i % block_size

            slot_mapping = []
            for block_table_i in block_table:
                slot = block_table_i[-1] * block_size + block_offset
                slot_mapping.append([slot])
            block_table_tensor = torch.tensor(block_table, dtype=torch.int64)
            slot_mapping_tensor = torch.tensor(slot_mapping, dtype=torch.int64)
            partial_page_tkv_mask = partial_page_tkv_mask + 1
            # batch
            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(block_table_tensor, 0)
            torch._dynamo.mark_dynamic(slot_mapping_tensor, 0)
            torch._dynamo.mark_dynamic(kwargs["position_ids"], 0)
            torch._dynamo.mark_dynamic(partial_page_tkv_mask, 0)
            torch._dynamo.mark_dynamic(left_padded_prompt_mask, 0)

            # seq
            torch._dynamo.mark_static(input_ids, 1)  # always 1
            torch._dynamo.mark_dynamic(block_table_tensor, 1)
            torch._dynamo.mark_static(slot_mapping_tensor, 1)  # always 1
            torch._dynamo.mark_static(kwargs["position_ids"], 1)  # always 1
            kwargs["custom_attention_op"] = PagedAttentionOp.from_decode_meta(
                slot_mapping_tensor, block_table_tensor, partial_page_tkv_mask, left_padded_prompt_mask, model.config.attention_multiplier
            )
        return input_ids, kwargs
    return _inner