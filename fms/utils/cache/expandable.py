from typing import Tuple, List, Optional, Union, Dict

from fms.utils.cache import CacheDataLayer, CacheDataWithMetadata, KVCacheManager
import dataclasses
import torch


@dataclasses.dataclass
class InPlaceCacheDataLayer(CacheDataLayer):
    data_layer: Tuple[torch.Tensor, torch.Tensor]

    def store(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = keys.shape
        self.data_layer[0][:, :, -shape[2] :, :].copy_(keys)
        self.data_layer[1][:, :, -shape[2] :, :].copy_(values)
        keys = self.data_layer[0]
        values = self.data_layer[1]
        return keys, values


@dataclasses.dataclass
class InPlaceCacheData(CacheDataWithMetadata):
    data: List[Tuple[torch.Tensor, torch.Tensor]]
    sequence_ids: List[int]
    max_sequence_length: int
    context_lengths: torch.Tensor
    is_generating: bool

    def get_layer(self, layer_index: int) -> InPlaceCacheDataLayer:
        return InPlaceCacheDataLayer(
            data_layer=self.data[layer_index],
        )

    def is_filled(self) -> bool:
        return self.is_generating


class ExpandableKVCacheManager(KVCacheManager):
    # TODO: Would be nice for this cache to be the compact expandable cache, but not required right now
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        emb_dim: int,
        tensor_parallel_size: int = 1,
        device: Optional[Union[str, torch.device]] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.num_layers = num_layers
        self.num_heads = (
            num_heads // tensor_parallel_size if num_heads > 1 else num_heads
        )
        self.head_size = emb_dim // num_heads
        self.device = device
        self.dtype = dtype
        self.context_map: Dict[int, int] = {}

    def allocate_prompt_tokens(
        self, num_tokens_per_sequence: List[int]
    ) -> InPlaceCacheData:
        # TODO: we might be able to handle multiple batches by using sequence ids, but for now naiive approach, one cache per batch
        self.cache.clear()
        self.context_map = {
            i: num_tokens for i, num_tokens in enumerate(num_tokens_per_sequence)
        }

        max_sequence_length = max(num_tokens_per_sequence)
        context_lengths = torch.tensor(num_tokens_per_sequence, dtype=torch.int32)
        sequence_ids = [i for i in range(len(num_tokens_per_sequence))]
        for i in range(self.num_layers):
            # b x head x sequence_length x emb_dim/head
            empty_tensor_k = torch.empty(
                size=(
                    len(num_tokens_per_sequence),
                    self.num_heads,
                    max_sequence_length,
                    self.head_size,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            empty_tensor_v = torch.empty(
                size=(
                    len(num_tokens_per_sequence),
                    self.num_heads,
                    max_sequence_length,
                    self.head_size,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            self.cache.append((empty_tensor_k, empty_tensor_v))

        return InPlaceCacheData(
            data=self.cache,
            context_lengths=context_lengths,
            max_sequence_length=max_sequence_length,
            sequence_ids=sequence_ids,
            is_generating=False,
        )

    def allocate_generated_tokens(
        self, sequence_ids: List[int], num_tokens_per_sequence: List[int]
    ) -> InPlaceCacheData:
        max_sequence_length = -1
        context_lengths = []
        for sequence_id, num_tokens in zip(sequence_ids, num_tokens_per_sequence):
            self.context_map[sequence_id] = self.context_map[sequence_id] + num_tokens
            max_sequence_length = max(
                max_sequence_length, self.context_map[sequence_id]
            )
            context_lengths.append(self.context_map[sequence_id])

        for i in range(self.num_layers):
            empty_tensor_k = torch.empty(
                size=(
                    len(num_tokens_per_sequence),
                    self.num_heads,
                    max(num_tokens_per_sequence),
                    self.head_size,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            empty_tensor_v = torch.empty(
                size=(
                    len(num_tokens_per_sequence),
                    self.num_heads,
                    max(num_tokens_per_sequence),
                    self.head_size,
                ),
                dtype=self.dtype,
                device=self.device,
            )
            self.cache[i] = (
                torch.cat((self.cache[i][0], empty_tensor_k), dim=2),
                torch.cat((self.cache[i][1], empty_tensor_v), dim=2),
            )

        context_lengths_tensor = torch.tensor(
            context_lengths, dtype=torch.int32, device=self.device
        )
        return InPlaceCacheData(
            data=self.cache,
            context_lengths=context_lengths_tensor,
            max_sequence_length=max_sequence_length,
            sequence_ids=sequence_ids,
            is_generating=True,
        )
