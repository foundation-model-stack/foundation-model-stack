import abc
import dataclasses
import queue
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch._inductor.ir as ir
import torch._inductor.lowering as lowering
from torch._inductor.virtualized import V
from fms._C import cache_ops, ops  # type: ignore

from fms.utils.cache import (
    CacheDataLayer,
    CacheDataWithMetadata,
    KVCacheManager,
    KVCache,
)

lib = torch.library.Library("paged_attention", "FRAGMENT")

lib.define(
    "reshape_and_cache(Tensor key, Tensor value, Tensor key_cache, Tensor value_cache, Tensor slot_mapping) -> (Tensor, Tensor)"
)

# needed for compile
@torch.library.impl(lib, "reshape_and_cache", "Meta")
def _reshape_and_cache_meta(key, value, key_cache, value_cache, slot_mapping):
    return key_cache.contiguous(), value_cache.contiguous()


@torch.library.impl(lib, "reshape_and_cache", "CUDA")
def _reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    key = key.contiguous()
    value = value.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    slot_mapping = slot_mapping.contiguous()
    cache_ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    return key_cache, value_cache


lowering.fallbacks.add(torch.ops.paged_attention.reshape_and_cache)


@lowering.register_lowering(
    torch.ops.paged_attention.reshape_and_cache, type_promotion_kind=None
)
def _reshape_and_cache_lowering(key, value, key_cache, value_cache, slot_mapping):
    PagedAttnKernel.create(
        torch.ops.paged_attention.reshape_and_cache.default,
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        mutated_inputs=[key_cache, value_cache],
    )
    return key_cache, value_cache


lib.define(
    "paged_attention_v1(Tensor out, Tensor query, Tensor key_cache, Tensor value_cache, Tensor head_mapping, float scale, Tensor block_tables, Tensor context_lens, int block_size, SymInt max_context_len, Tensor? alibi_slopes) -> Tensor"
)


@torch.library.impl(lib, "paged_attention_v1", "Meta")
def _paged_attention_v1_meta(
    out,
    query,
    key_cache,
    value_cache,
    head_mapping,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    return out.contiguous()


@torch.library.impl(lib, "paged_attention_v1", "CUDA")
def _paged_attention_v1(
    out,
    query,
    key_cache,
    value_cache,
    head_mapping,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    out = out.contiguous()
    query = query.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    head_mapping = head_mapping.contiguous()
    block_tables = block_tables.contiguous()
    context_lens = context_lens.contiguous()

    ops.paged_attention_v1(
        out,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    )
    return out


lowering.fallbacks.add(torch.ops.paged_attention.paged_attention_v1)


@lowering.register_lowering(
    torch.ops.paged_attention.paged_attention_v1, type_promotion_kind=None
)
def _paged_attention_v1_lowering(
    out,
    query,
    key_cache,
    value_cache,
    head_mapping,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    PagedAttnKernel.create(
        torch.ops.paged_attention.paged_attention_v1.default,
        out,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        mutated_inputs=[out],
    )
    return out


# Available starting PT 2.2
class NoneLayout(ir.IRNode):
    def __init__(self, device):
        self.device = device
        self.size = [0]
        self.stride = [0]

    def storage_size(self):
        return 0

    def as_fixed(self):
        return self


# Available starting PT 2.2
class MutationOutput(ir.ExternKernel):
    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def __init__(self, layout, input, parent):
        super().__init__(None, layout, [input, parent], ())
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        return False

    def is_no_op(self):
        return True

    def has_side_effects(self):
        return True

    def get_alias_names(self):
        return [self.inputs[0].get_name()]


class PagedAttnKernel(ir.FallbackKernel):
    def should_allocate(self):
        return False

    def has_side_effects(self):
        return True

    @classmethod
    def create(cls, kernel, *args, mutated_inputs=[], **kwargs) -> None:
        with V.graph.fake_mode:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                schema,
            ) = cls.process_kernel(kernel, *args, **kwargs)
        for tensor_arg in tensor_args:
            tensor_arg.realize()

        packed = cls(
            NoneLayout(tensor_args[0].get_device()),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            schema=schema,
        )
        # Mark inplace inputs as mutated
        for kernel_input in mutated_inputs:
            V.graph.mark_buffer_mutated(kernel_input.get_name())
            MutationOutput(kernel_input.layout, kernel_input, packed)


@dataclasses.dataclass
class PagedAttentionCacheDataLayer(CacheDataLayer):
    data_layer: Tuple[torch.Tensor, torch.Tensor]
    max_sequence_length: int
    context_lengths: torch.Tensor
    slot_mapping: torch.Tensor
    block_mapping: torch.Tensor
    block_size: int
    num_heads: int  # this could be kvheads or num_heads
    head_size: int
    is_generating: bool

    def store(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_to_cache = keys.transpose(2, 1).reshape(-1, self.num_heads, self.head_size)
        value_to_cache = values.transpose(2, 1).reshape(
            -1, self.num_heads, self.head_size
        )

        (
            keys_cache_output,
            values_cache_output,
        ) = torch.ops.paged_attention.reshape_and_cache(
            key_to_cache,
            value_to_cache,
            self.data_layer[0],
            self.data_layer[1],
            self.slot_mapping,
        )

        if self.is_generating:
            return keys_cache_output, values_cache_output
        else:
            return keys, values


@dataclasses.dataclass
class PagedAttentionCacheData(CacheDataWithMetadata):
    data: List[Tuple[torch.Tensor, torch.Tensor]]
    max_sequence_length: int
    context_lengths: torch.Tensor
    slot_mapping: torch.Tensor
    block_mapping: torch.Tensor
    block_size: int
    num_heads: int  # this could be kvheads or num_heads
    head_size: int
    is_generating: bool
    sequence_ids: List[int]

    def get_layer(self, layer_index: int) -> PagedAttentionCacheDataLayer:
        return PagedAttentionCacheDataLayer(
            data_layer=self.data[layer_index],
            max_sequence_length=self.max_sequence_length,
            context_lengths=self.context_lengths,
            slot_mapping=self.slot_mapping,
            block_mapping=self.block_mapping,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_size=self.head_size,
            is_generating=self.is_generating,
        )

    def is_filled(self) -> bool:
        return self.is_generating


def get_cache_block_size(block_size, head_size, num_heads, num_layers, dtype) -> int:
    kv_cache_block_size = block_size * num_heads * head_size * 2  # 2 for k and v

    total_size = num_layers * kv_cache_block_size
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    return dtype_size * total_size


def get_max_gpu_blocks_available(
    block_size: int,
    emb_dim: int,
    nheads: int,
    nlayers: int,
    gpu_memory_utilization: float,
    dtype,
) -> int:

    # Calculate the number of blocks that can be allocated with the
    # profiled peak memory.
    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated()
    total_gpu_memory = torch.cuda.get_device_properties("cuda").total_memory
    cache_block_size = get_cache_block_size(
        block_size, emb_dim // nheads, nheads, nlayers, dtype
    )
    num_gpu_blocks = int(
        (total_gpu_memory * gpu_memory_utilization - peak_memory) // cache_block_size
    )
    num_gpu_blocks = max(num_gpu_blocks, 0)
    torch.cuda.empty_cache()
    return num_gpu_blocks


class CacheBlock:
    def __init__(
        self,
        block_number: int,
        block_size: int,
    ):
        self.block_number = block_number
        self.block_size = block_size
        self.num_tokens = 0

    def num_available_slots(self) -> int:
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        return self.num_available_slots() == 0

    def append_num_tokens(self, num_tokens: int):
        # todo: we need some way of differentiating number of tokens stored in the cache vs num allocated
        self.num_tokens += num_tokens

    def subtract_num_tokens(self, num_tokens: int):
        self.num_tokens -= num_tokens

    def __repr__(self):
        return f"CacheBlock(block_number={self.block_number}, block_size={self.block_size}, num_tokens={self.num_tokens})"


class CacheBlockGroup(List[CacheBlock]):
    def __init__(self, sequence_id: int, block_size: int):
        super().__init__()
        self.sequence_id = sequence_id
        self.block_size = block_size
        self._is_generating = False
        self._is_initialized_with_prompt = False
        self.prefix: Optional[CacheBlockGroup] = None
        self.ref_count = 0

    @classmethod
    def from_prefix(cls, sequence_id: int, prefix: "CacheBlockGroup"):
        cbg = cls(sequence_id, prefix.block_size)
        cbg._is_generating = True
        cbg._is_initialized_with_prompt = True

        # add duplicate blocks
        for cb in prefix:
            cbg.append(cb)

        # set the prefix
        cbg.prefix = prefix
        # update the reference count of the prefix
        prefix.ref_count += 1

        return cbg

    def remove_tokens(self, num_tokens: int) -> List[CacheBlock]:
        # remove tokens and return the blocks to be freed
        if num_tokens > list.__len__(self):
            raise ValueError(
                "the number of tokens to remove is greater than what exists in this cache block group not including the prefix"
            )
        num_tokens_to_remove = num_tokens
        blocks_to_free = []
        for cb in reversed(self):
            if cb.num_tokens < num_tokens_to_remove:
                num_tokens_to_remove -= cb.num_tokens
                cb.num_tokens = 0
                blocks_to_free.append(cb)
                num_tokens_to_remove -= cb.num_tokens
            else:
                # if its equal to num tokens, we don't need to free the block as it can just be re-used
                cb.subtract_num_tokens(num_tokens_to_remove)
                break
        # remove the blocks from this CacheBlockGroup that are to be freed in the cache manager
        for _ in blocks_to_free:
            self.pop()
        return blocks_to_free

    def is_initialized_with_prompt(self):
        return self._is_initialized_with_prompt

    def is_generating(self):
        return self._is_generating

    def last_cache_block_is_full(self):
        return self[-1].is_full()

    def get_sequence_length(self):
        if len(self) == 0:
            return 0
        else:
            return sum([cb.num_tokens for cb in self])

    def get_cache_block(self, position: int):
        return self[position // self.block_size]

    def get_slot_mapping(self, position: Optional[int] = None) -> List[int]:
        slot_mapping = []
        start = position if position else 0
        for position_i in range(start, self.get_sequence_length()):
            block_number = self.get_cache_block(position_i).block_number
            block_offset = position_i % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append(slot)
        return slot_mapping

    def get_block_mapping(self):
        return [cb.block_number for cb in self]


class PagedKVCacheManager(KVCacheManager):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        emb_dim: int,
        total_num_gpu_blocks: Optional[int] = None,
        block_size: int = 16,
        tensor_parallel_size: int = 1,
        device: Optional[Union[str, torch.device]] = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.block_size = block_size
        self.cache: List[KVCache] = []
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.device = device
        self.num_heads = (
            num_heads // tensor_parallel_size if num_heads > 1 else num_heads
        )

        if not total_num_gpu_blocks:
            total_num_gpu_blocks = get_max_gpu_blocks_available(
                block_size,
                emb_dim,
                num_heads // tensor_parallel_size if num_heads > 1 else num_heads,
                num_layers,
                0.8,
                dtype,
            )
        self.total_num_gpu_blocks = total_num_gpu_blocks

        self.head_size = emb_dim // num_heads

        x = self.block_size // element_size
        key_block_shape = (
            self.num_heads,
            self.head_size // x,
            block_size,
            x,
        )
        value_block_shape = (
            self.num_heads,
            self.head_size,
            block_size,
        )
        for _ in range(num_layers):
            key_blocks = torch.empty(
                size=(total_num_gpu_blocks, *key_block_shape),
                dtype=dtype,
                device=self.device,
            )
            value_blocks = torch.empty(
                size=(total_num_gpu_blocks, *value_block_shape),
                dtype=dtype,
                device=self.device,
            )
            self.cache.append((key_blocks, value_blocks))

        self.free_blocks: List[CacheBlock] = []
        self.unused_keys: queue.Queue[int] = queue.Queue(len(self.free_blocks))
        for i in range(total_num_gpu_blocks):
            self.free_blocks.append(CacheBlock(i, block_size))
            self.unused_keys.put_nowait(i)

        # each sequence will be mapped to a cache block group
        # for now this will just assume we always have the same sequences in batch
        self.cbg_map: Dict[int, CacheBlockGroup] = {}

    def get_max_sequence_length(self, sequence_ids: List[int]) -> int:
        return max(
            [self.cbg_map[seq_id].get_sequence_length() for seq_id in sequence_ids]
        )

    def _allocate_block(self) -> CacheBlock:
        return self.free_blocks.pop()

    @staticmethod
    def __pad_to_max_left(x: List[int], max_len: int, pad: int) -> List[int]:
        return [pad] * (max_len - len(x)) + x

    @staticmethod
    def __pad_to_max_right(x: List[int], max_len: int, pad: int) -> List[int]:
        return x + [pad] * (max_len - len(x))

    def is_generating(self, sequence_ids: List[int]):
        for sequence_id in sequence_ids:
            if (
                sequence_id not in self.cbg_map
                or not self.cbg_map[sequence_id].is_generating()
            ):
                return False
        return True

    def is_initialized_with_prompt(self, sequence_ids: List[int]):
        for sequence_id in sequence_ids:
            if (
                sequence_id not in self.cbg_map
                or not self.cbg_map[sequence_id].is_initialized_with_prompt()
            ):
                return False
        return True

    def free(self, sequence_id: int):
        if sequence_id not in self.cbg_map:
            return
        cbg = self.cbg_map[sequence_id]

        if cbg.ref_count != 0:
            raise ValueError(
                f"This sequence id is being reference by other sequences and cannot be freed"
            )

        # remove a reference count from all cache block groups that was a prefix as part of this sequence
        if cbg.prefix is not None:
            cbg.prefix.ref_count -= 1
            prefix_block_numbers = set(cbg.prefix.get_block_mapping())

        for cb in cbg:
            if cbg.prefix is None or (
                cbg.prefix is not None and cb.block_number not in prefix_block_numbers
            ):
                cb.num_tokens = 0
                self.free_blocks.append(cb)
        self.unused_keys.put_nowait(sequence_id)
        del self.cbg_map[sequence_id]

    def free_sequences(self, sequence_ids: List[int]):
        for seq_id in sequence_ids:
            self.free(seq_id)

    def _get_unassigned_sequence_id(self) -> int:
        return self._get_unassigned_sequence_ids(1)[0]

    def _get_unassigned_sequence_ids(self, num_sequences: int) -> List[int]:
        return [self.unused_keys.get_nowait() for _ in range(num_sequences)]

    def _get_cache_metadata(
        self,
        sequence_ids: List[int],
        is_prompt: bool,
        num_tokens_per_sequence: Optional[List[int]] = None,
    ) -> PagedAttentionCacheData:
        slot_mapping = []
        block_tables = []
        context_lengths = []
        max_sequence_length = self.get_max_sequence_length(sequence_ids)
        remainder = max_sequence_length % self.block_size
        max_num_blocks = max_sequence_length // self.block_size
        if remainder != 0:
            max_num_blocks += 1
        i = 0
        for sequence_id in sequence_ids:
            cbg = self.cbg_map[sequence_id]

            context_length = cbg.get_sequence_length()
            if is_prompt:
                slot = cbg.get_slot_mapping()
                slot = self.__pad_to_max_left(slot, max_sequence_length, -1)
            else:
                num_tokens = num_tokens_per_sequence[i]  # type: ignore
                start = context_length - num_tokens
                slot = cbg.get_slot_mapping(start)
                i += 1

            block_mapping = cbg.get_block_mapping()
            block_mapping = self.__pad_to_max_right(block_mapping, max_num_blocks, 0)

            slot_mapping.append(slot)
            block_tables.append(block_mapping)
            context_lengths.append(context_length)

        slot_mapping_tensor = torch.tensor(
            slot_mapping, dtype=torch.long, device=self.device
        )
        block_tables_tensor = torch.tensor(
            block_tables, dtype=torch.int, device=self.device
        )
        context_lengths_tensor = torch.tensor(
            context_lengths, dtype=torch.int, device=self.device
        )

        return PagedAttentionCacheData(
            data=self.cache,
            max_sequence_length=max_sequence_length,
            context_lengths=context_lengths_tensor,
            slot_mapping=slot_mapping_tensor,
            block_mapping=block_tables_tensor,
            block_size=self.block_size,
            num_heads=self.num_heads,
            head_size=self.head_size,
            is_generating=not is_prompt,
            sequence_ids=sequence_ids,
        )

    def allocate_prompt_tokens(
        self, num_tokens_per_sequence: List[int]
    ) -> PagedAttentionCacheData:
        sequence_ids = self._get_unassigned_sequence_ids(len(num_tokens_per_sequence))

        for seq_id, num_tokens in zip(sequence_ids, num_tokens_per_sequence):
            self._allocate_prompt_sequence(seq_id, num_tokens)

        return self._get_cache_metadata(sequence_ids, is_prompt=True)

    def allocate_generated_tokens(
        self, sequence_ids: List[int], num_tokens_per_sequence: List[int]
    ) -> PagedAttentionCacheData:
        for seq_id, num_tokens in zip(sequence_ids, num_tokens_per_sequence):
            cache_block_group = self.cbg_map[seq_id]
            cache_block_group._is_generating = True

            for i in range(num_tokens):
                if cache_block_group.last_cache_block_is_full():
                    last_block = self._allocate_block()
                    last_block.append_num_tokens(1)
                    cache_block_group.append(last_block)
                else:
                    cache_block_group[-1].append_num_tokens(1)

        return self._get_cache_metadata(
            sequence_ids,
            is_prompt=False,
            num_tokens_per_sequence=num_tokens_per_sequence,
        )

    def _allocate_prompt_sequence(self, seq_id: int, num_tokens: int):
        cache_block_group: CacheBlockGroup = CacheBlockGroup(seq_id, self.block_size)

        # one block allocation will happen automatically as the group always starts empty
        last_cache_block = self._allocate_block()

        cursor = 0
        while cursor < num_tokens:
            tokens_to_append = (
                min(num_tokens, cursor + last_cache_block.num_available_slots())
                - cursor
            )
            last_cache_block.append_num_tokens(tokens_to_append)
            cursor += tokens_to_append

            if cursor >= num_tokens:
                # we are done, so we need to append but not allocate
                cache_block_group.append(last_cache_block)
            elif last_cache_block.is_full():
                # if the block is full we can append it
                cache_block_group.append(last_cache_block)
                # because the other condition did not hold, we can allocate a new block
                last_cache_block = self._allocate_block()

        cache_block_group._is_initialized_with_prompt = True
        self.cbg_map[seq_id] = cache_block_group

    def add_child_sequence(self, parent_sequence_id: int) -> int:
        parent_cbg = self.cbg_map[parent_sequence_id]

        child_sequence_id = self._get_unassigned_sequence_id()
        child_cbg = CacheBlockGroup.from_prefix(child_sequence_id, parent_cbg)
        key_caches = [key_cache for key_cache, _ in self.cache]
        value_caches = [value_cache for _, value_cache in self.cache]

        if not parent_cbg.last_cache_block_is_full():
            new_block_to_copy = self._allocate_block()
            cache_ops.copy_blocks(
                key_caches,
                value_caches,
                {parent_cbg[-1].block_number: [new_block_to_copy.block_number]},
            )
            new_block_to_copy.append_num_tokens(parent_cbg[-1].num_tokens)
            child_cbg.pop()
            child_cbg.append(new_block_to_copy)

        self.cbg_map[child_sequence_id] = child_cbg

        return child_sequence_id

    def add_child_sequences(
        self, parent_sequence_id: int, num_sequences: int
    ) -> list[int]:
        child_sequence_ids = []
        for _ in range(num_sequences):
            child_sequence_ids.append(self.add_child_sequence(parent_sequence_id))
        return child_sequence_ids

    def remove_tokens(self, sequence_id: int, num_tokens: int):
        blocks_to_free = self.cbg_map[sequence_id].remove_tokens(num_tokens)
        for cb in blocks_to_free:
            self.free_blocks.append(cb)
