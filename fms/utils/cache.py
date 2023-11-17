from typing import Tuple, List, Dict, Optional

import torch

KVCache = Tuple[torch.Tensor, torch.Tensor]  # (key cache, value cache)


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
        self.num_tokens += num_tokens


class CacheBlockGroup(List[CacheBlock]):
    def __init__(self, block_size: int):
        super().__init__()
        self.block_size = block_size

    def __getitem__(self, key):
        return super(CacheBlockGroup, self).__getitem__(key)

    def last_cache_block_is_full(self):
        return self[-1].is_full()

    def get_sequence_length(self):
        return sum([cb.num_tokens for cb in self])

    def get_cache_block(self, position: int):
        return self[position // self.block_size]

    def get_slot_mapping(self, position: Optional[int] = None) -> List[int]:
        slot_mapping = []
        start = position if position else 0
        for position_i in range(start, self.get_sequence_length()):
            block_number = self.get_cache_block(position)
            block_offset = position % self.block_size
            slot = block_number * self.block_size + block_offset
            slot_mapping.append(slot)
        return slot_mapping


class PagedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        emb_dim: int,
        total_num_gpu_blocks: Optional[int] = None,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
    ):
        self.block_size = block_size
        self.cache: List[KVCache] = []
        element_size = torch.tensor([], dtype=dtype).element_size()

        if not total_num_gpu_blocks:
            total_num_gpu_blocks = get_max_gpu_blocks_available(
                block_size, emb_dim, num_heads, num_layers, 0.8, dtype
            )

        head_size = emb_dim // num_heads

        x = 16 // element_size
        key_block_shape = (
            num_heads,
            head_size // x,
            block_size,
            x,
        )
        value_block_shape = (
            num_heads,
            head_size,
            block_size,
        )
        for _ in range(num_layers):
            key_blocks = torch.empty(
                size=(total_num_gpu_blocks, *key_block_shape),
                dtype=dtype,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(total_num_gpu_blocks, *value_block_shape),
                dtype=dtype,
                device="cuda",
            )
            self.cache.append((key_blocks, value_blocks))

        self.free_blocks: List[CacheBlock] = []

        for i in range(total_num_gpu_blocks):
            self.free_blocks.append(CacheBlock(i, block_size))

        # each sequence will be mapped to a cache block group
        # for now this will just assume we always have the same sequences in batch
        self.block_table_map: Dict[int, CacheBlockGroup] = {}
        self.cache_empty = True

    def get_max_sequence_length(self):
        return max([cb.get_sequence_length() for cb in self.block_table_map.values()])

    def _allocate_block(self) -> CacheBlock:
        return self.free_blocks.pop(0)

    def is_empty(self):
        return self.cache_empty

    def allocate_initial_prompt(self, prompt_tensor: torch.Tensor):
        prompt_list = prompt_tensor.tolist()
        for i, seq_id in enumerate(prompt_list):
            self._allocate_prompt_sequence(i, seq_id)

    def allocate_generated_token(self, generated_tensor: torch.Tensor):
        for i in range(generated_tensor.size(0)):
            cache_block_group = self.block_table_map[i]

            if cache_block_group.last_cache_block_is_full():
                last_block = self._allocate_block()
                last_block.append_num_tokens(1)
                cache_block_group.append(last_block)
        self.cache_empty = False

    def _allocate_prompt_sequence(self, seq_id: int, tokens: List[int]):
        cache_block_group: CacheBlockGroup = CacheBlockGroup()

        # one block allocation will happen automatically as the group always starts empty
        last_cache_block = self._allocate_block()

        cursor = 0
        while cursor < len(tokens):
            tokens_to_append = (
                min(len(tokens), cursor + last_cache_block.num_available_slots())
                - cursor
            )
            last_cache_block.append_num_tokens(tokens_to_append)
            cursor += tokens_to_append

            if cursor >= len(tokens):
                # we are done, so we need to append but not allocate
                cache_block_group.append(last_cache_block)
            elif last_cache_block.is_full():
                # if the block is full we can append it
                cache_block_group.append(last_cache_block)
                # because the other condition did not hold, we can allocate a new block
                last_cache_block = self._allocate_block()

        self.block_table_map[seq_id] = cache_block_group
