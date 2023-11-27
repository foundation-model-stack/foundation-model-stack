import itertools
from abc import ABCMeta, abstractmethod
from typing import List, Type

import torch
import torch.nn as nn
from torch.distributed.distributed_c10d import ProcessGroup

from fms.distributed.tensorparallel import (
    apply_colwise_tp,
    apply_embedding_tp,
    apply_rowwise_tp,
)


class TPModule(nn.Module, metaclass=ABCMeta):
    rank: int
    world_size: int

    def setup_tp(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def list_colwise_weights(self) -> List[str]:
        return []

    @abstractmethod
    def list_rowwise_weights(self) -> List[str]:
        return []

    @abstractmethod
    def list_embedding_weights(self) -> List[str]:
        return []

    @staticmethod
    @abstractmethod
    def import_module(module, group: ProcessGroup):
        pass

    def import_weights(self, module: nn.Module):
        for weight in self.list_colwise_weights():
            apply_colwise_tp(
                getattr(self, weight),
                getattr(module, weight),
                self.world_size,
                self.rank,
            )
        for weight in self.list_rowwise_weights():
            apply_rowwise_tp(
                getattr(self, weight),
                getattr(module, weight),
                self.world_size,
                self.rank,
            )
        for weight in self.list_embedding_weights():
            apply_embedding_tp(
                getattr(self, weight),
                getattr(module, weight),
                self.world_size,
                self.rank,
            )
        tp_sharded_modules = list(
            itertools.chain(
                self.list_colwise_weights(),
                self.list_rowwise_weights(),
                self.list_embedding_weights(),
            )
        )
        with torch.no_grad():
            for mod_name, module in self.named_children():
                if not mod_name in tp_sharded_modules:
                    for param_name, param in module.named_parameters(recurse=False):
                        param.copy_(
                            getattr(getattr(module, mod_name), param_name),
                            non_blocking=True,
                        )
