import abc
from typing import List, Tuple, Optional

import torch


class KVCacheUnit(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_cache_unit(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def append(self, k: torch.Tensor, v: torch.Tensor):
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class DynamicKVCacheUnit(KVCacheUnit):
    def __init__(self):
        self.cache = None

    def get_cache_unit(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cache

    def append(self, k: torch.Tensor, v: torch.Tensor):
        if self.cache is None:
            self.cache = (k, v)
        else:
            self.cache = (
                torch.concat((self.cache[0], k), dim=2),
                torch.concat((self.cache[1], v), dim=2),
            )

    def __len__(self) -> int:
        if self.cache is None:
            return 0
        else:
            return self.cache[0][0].size(-2)


class PreAllocatedKVCacheUnit(KVCacheUnit):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        batch_size: int,
        max_length: int,
    ):
        self.len = 0
        self.cache = (
            # k
            torch.empty(
                batch_size,
                num_heads,
                max_length,
                emb_dim // num_heads,
            ),
            # v
            torch.empty(
                batch_size,
                num_heads,
                max_length,
                emb_dim // num_heads,
            ),
        )

    def append(self, k: torch.Tensor, v: torch.Tensor):
        self.cache[0][:, :, self.len : self.len + 1, :].copy_(k)
        self.cache[1][:, :, self.len : self.len + 1, :].copy_(v)
        self.len += k.size(-2)

    def get_cache_unit(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cache[0][:, : self.len, :], self.cache[1][:, : self.len, :]

    def __len__(self):
        return self.len


cache = PreAllocatedKVCacheUnit(10, 2, 1, 15)

print(cache.get_cache_unit())
for i in range(15):
    cache.append(torch.zeros(1, 2, 1, 5) + i, torch.zeros(1, 2, 1, 5) + i + 1)

print(cache.get_cache_unit())
