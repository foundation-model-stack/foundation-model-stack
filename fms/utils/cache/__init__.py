import abc
import dataclasses
from typing import Tuple, List, Optional
import torch


@dataclasses.dataclass
class CacheDataLayer(metaclass=abc.ABCMeta):
    data_layer: Tuple[torch.Tensor, torch.Tensor]

    @abc.abstractmethod
    def store(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


@dataclasses.dataclass
class CacheData(metaclass=abc.ABCMeta):
    data: List[Tuple[torch.Tensor, torch.Tensor]]

    @abc.abstractmethod
    def get_layer(self, layer_index: int) -> CacheDataLayer:
        pass

    @abc.abstractmethod
    def is_filled(self) -> bool:
        pass


@dataclasses.dataclass
class CacheDataWithMetadata(CacheData):
    data: List[Tuple[torch.Tensor, torch.Tensor]]
    sequence_ids: List[int]
    max_sequence_length: int
    context_lengths: torch.Tensor


class KVCacheManager(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def allocate_prompt_tokens(
        self, num_tokens_per_sequence: List[int]
    ) -> CacheDataWithMetadata:
        pass

    @abc.abstractmethod
    def allocate_generated_tokens(
        self, sequence_ids: List[int], num_tokens_per_sequence: List[int]
    ) -> CacheDataWithMetadata:
        pass


KVCache = Tuple[torch.Tensor, torch.Tensor]  # (key cache, value cache)


@dataclasses.dataclass
class OutOfPlaceCacheDataLayer(CacheDataLayer):
    data_layer: Tuple[torch.Tensor, torch.Tensor]

    def store(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data_layer is not None:
            self.data_layer = (
                torch.cat((self.data_layer[0], keys), dim=2),
                torch.cat((self.data_layer[1], values), dim=2),
            )
            keys, values = self.data_layer
        return keys, values


class OutOfPlaceCacheData(CacheData):
    def __init__(self, data: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.data = data

    def get_layer(self, layer_index: int) -> OutOfPlaceCacheDataLayer:
        return OutOfPlaceCacheDataLayer(data_layer=self.data[layer_index])

    def is_filled(self) -> bool:
        return self.data[0] is not None
