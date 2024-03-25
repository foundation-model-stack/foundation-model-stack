import itertools
from abc import ABCMeta, abstractmethod
from typing import List, Type

import torch
import torch.nn as nn
from torch.distributed.distributed_c10d import ProcessGroup

from fms.distributed.tensorparallel import (
    apply_colwise_tp,
    apply_embedding_tp,
    apply_moe_tp,
    apply_rowwise_tp,
)


def _get_tpd_module(module: nn.Module, attr_name: str):
    if attr_name == "self":
        return module
    return getattr(module, attr_name)


class TPModule(nn.Module, metaclass=ABCMeta):
    """
    This is an abstract class that any nn.Module can implement to enable
    Tensor Parallel. On top of inheriting from this class, the TP module
    will have to implement list_colwise_weights, list_rowwise_weights,
    list_embedding_weights, and import_module for their relevant weights.
    Finally, the module must call setup_tp at the end of their __init__
    function. See examples in attention.py, feedforward.py and embedding.py

    """

    rank: int
    world_size: int

    def setup_tp(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size

    def colwise_param_names(self) -> List[str]:
        return []

    def rowwise_param_names(self) -> List[str]:
        return []

    def embedding_param_names(self) -> List[str]:
        return []

    def moe_param_names(self) -> List[str]:
        return []

    @staticmethod
    @abstractmethod
    def import_module(module, group: ProcessGroup):
        pass

    def import_weights(self, module: nn.Module):
        for weight in self.colwise_param_names():
            apply_colwise_tp(
                _get_tpd_module(self, weight),
                _get_tpd_module(module, weight),
                self.rank,
                self.world_size,
            )
        for weight in self.rowwise_param_names():
            apply_rowwise_tp(
                _get_tpd_module(self, weight),
                _get_tpd_module(module, weight),
                self.rank,
                self.world_size,
            )
        for weight in self.embedding_param_names():
            apply_embedding_tp(
                _get_tpd_module(self, weight),
                _get_tpd_module(module, weight),
                self.rank,
                self.world_size,
            )
        if len(self.moe_param_names()) > 0:
            apply_moe_tp(
                self,
                module,
                self.moe_param_names(),
                self.rank,
                self.world_size,
            )
        tp_sharded_modules = list(
            itertools.chain(
                self.colwise_param_names(),
                self.rowwise_param_names(),
                self.embedding_param_names(),
                self.moe_param_names(),
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
