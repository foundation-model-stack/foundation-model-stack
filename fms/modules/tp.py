import itertools
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Type

import torch
import torch.nn as nn
from torch.distributed.distributed_c10d import ProcessGroup

from fms.distributed.tensorparallel import (
    apply_colwise_tp,
    apply_embedding_tp,
    apply_rowwise_tp,
)


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

    def colwise_params(self) -> Dict[str, List[int]]:
        """Override this method to mark weights as column-wise tensor-parallel. This method will also decide for each
        weight, how the weight is to be split. Each weight name will have a List[int] (max_partition_sizes) associated
        with it where if the list is larger than size 1, this implies the weight is fused (where the length of the list
        is the number of fused weights in a single parameter). The max_partition_sizes act as follows:

         for each number in the list, if world_size is smaller than or equal to that number, the tensor will get
        partitioned in worldsize parts, else if world size is larger than the number then you will get world size parts
        replicated in worldsize / number

        world_size = 4, max_partition_sizes = [8], tensor = [0 1 2 3 4 5 6 7]
        [0 1] [2 3] [4 5] [6 7]

        world_size = 8, max_partition_sizes = [4], tensor = [0 1 2 3 4 5 6 7]
        [0 1] [0 1] [2 3] [2 3] [4 5] [4 5] [6 7] [6 7]

        If there are multiple numbers in the max_partition_sizes list, then the param gets filled with non-contiguous
        slices of the tensor_value. This is useful for fused weight cases (qkv, mlp, moe, etc.)

        world_size = 4, max_partition_sizes = [4, 4], tensor = [0 1 2 3 4 5 6 7]
        [0 4] [1 5] [2 6] [3 7]

        world_size = 4, max_partition_sizes = [4, 1], tensor = [0 1 2 3 4 5 6 7 8 9]
        [0 1 8 9] [2 3 8 9] [4 5 8 9] [6 7 8 9]

        Returns:
        Dict[str, List[int]]
            a dictionary of weight names to their corresponding max_partition_sizes list
        """
        return {}

    def rowwise_params(self) -> Dict[str, List[int]]:
        """Override this method to mark weights as row-wise tensor-parallel. This method will also decide for each
        weight, how the weight is to be split. Each weight name will have a List[int] (max_partition_sizes) associated
        with it where if the list is larger than size 1, this implies the weight is fused (where the length of the list
        is the number of fused weights in a single parameter). The max_partition_sizes act as follows:

         for each number in the list, if world_size is smaller than or equal to that number, the tensor will get
        partitioned in worldsize parts, else if world size is larger than the number then you will get world size parts
        replicated in worldsize / number

        world_size = 4, max_partition_sizes = [8], tensor = [0 1 2 3 4 5 6 7]
        [0 1] [2 3] [4 5] [6 7]

        world_size = 8, max_partition_sizes = [4], tensor = [0 1 2 3 4 5 6 7]
        [0 1] [0 1] [2 3] [2 3] [4 5] [4 5] [6 7] [6 7]

        If there are multiple numbers in the max_partition_sizes list, then the param gets filled with non-contiguous
        slices of the tensor_value. This is useful for fused weight cases (qkv, mlp, moe, etc.)

        world_size = 4, max_partition_sizes = [4, 4], tensor = [0 1 2 3 4 5 6 7]
        [0 4] [1 5] [2 6] [3 7]

        world_size = 4, max_partition_sizes = [4, 1], tensor = [0 1 2 3 4 5 6 7 8 9]
        [0 1 8 9] [2 3 8 9] [4 5 8 9] [6 7 8 9]

        Returns:
        Dict[str, List[int]]
            a dictionary of weight names to their corresponding max_partition_sizes list
        """
        return {}

    def __get_start_index(
        self,
        tensor_value,
        output_size_per_partition,
        replication,
        max_partition_sizes,
        min_partition_size,
        cusum_max_partition_sizes,
        weight_i,
    ):
        return (self.rank // max(1, self.world_size // replication)) * (
            output_size_per_partition
            * (max_partition_sizes[weight_i] // min_partition_size)
        ) + (
            cusum_max_partition_sizes[weight_i]
            * tensor_value.shape[0]
            // cusum_max_partition_sizes[-1]
        )

    def __get_end_index(
        self,
        tensor_value,
        output_size_per_partition,
        replication,
        max_partition_sizes,
        min_partition_size,
        cusum_max_partition_sizes,
        weight_i,
    ):
        return ((self.rank // max(1, self.world_size // replication)) + 1) * (
            output_size_per_partition
            * (max_partition_sizes[weight_i] // min_partition_size)
        ) + (
            cusum_max_partition_sizes[weight_i]
            * tensor_value.shape[0]
            // cusum_max_partition_sizes[-1]
        )

    def _copy_rowwise(
        self,
        param: torch.nn.Parameter,
        tensor_value,
        is_bias,
        max_partition_sizes,
    ):
        """
        This function copies the correct shard of the weights for a rowwise-TP'd module
        according to the rank of the process and the world_size.

        Args
        ====
        param: torch.nn.Parameter
            Parameter that has had TP applied
        tensor_value: torch.Tensor
            tensor that needs sharding
        max_partition_sizes: List[int]
            for each number in the list, if world_size is smaller than or equal to that number, the tensor will get
            partitioned in worldsize parts, else if world size is larger than the number then you will get world size parts
            replicated in worldsize / number

            world_size = 4, max_partition_sizes = [8], tensor = [0 1 2 3 4 5 6 7]
            [0 1] [2 3] [4 5] [6 7]

            world_size = 8, max_partition_sizes = [4], tensor = [0 1 2 3 4 5 6 7]
            [0 1] [0 1] [2 3] [2 3] [4 5] [4 5] [6 7] [6 7]

            If there are multiple numbers in the max_partition_sizes list, then the param gets filled with non-contiguous
            slices of the tensor_value. This is useful for fused weight cases (qkv, mlp, moe, etc.)

            world_size = 4, max_partition_sizes = [4, 4], tensor = [0 1 2 3 4 5 6 7]
            [0 4] [1 5] [2 6] [3 7]

            world_size = 4, max_partition_sizes = [4, 1], tensor = [0 1 2 3 4 5 6 7 8 9]
            [0 1 8 9] [2 3 8 9] [4 5 8 9] [6 7 8 9]

        is_bias: bool
            if True is bias, else is weight
        rank: int
            Rank of the current process
        world_size: int
            Total number of TP processes
        """
        # Divide the weight matrix along the first dimension.
        cusum_max_partition_sizes = [0]
        min_partition_size = min(max_partition_sizes)
        for m in max_partition_sizes:
            cusum_max_partition_sizes.append(
                cusum_max_partition_sizes[-1] + (m // min_partition_size)
            )

        output_size_per_partition = param.shape[1] // (
            sum(max_partition_sizes) // min(max_partition_sizes)
        )
        if not is_bias:
            tensor = torch.cat(
                [
                    tensor_value[
                        :,
                        self.__get_start_index(
                            tensor_value,
                            output_size_per_partition,
                            replication,
                            max_partition_sizes,
                            min_partition_size,
                            cusum_max_partition_sizes,
                            i,
                        ) : self.__get_end_index(
                            tensor_value,
                            output_size_per_partition,
                            replication,
                            max_partition_sizes,
                            min_partition_size,
                            cusum_max_partition_sizes,
                            i,
                        ),
                    ]
                    for i, replication in enumerate(max_partition_sizes)
                ],
                dim=1,
            )
            param.copy_(tensor, non_blocking=True)
        else:
            if self.rank == 0:
                param.copy_(tensor_value, non_blocking=True)
            else:
                param.zero_()

    def _copy_colwise(
        self,
        param: torch.nn.Parameter,
        tensor_value,
        is_bias,
        max_partition_sizes,
    ):
        """
        This function copies the correct shard of the weights for a colwise-TP'd module
        according to the rank of the process and the world_size.

        Args
        ====
        param: torch.nn.Parameter
            Parameter that has had TP applied
        tensor_value: torch.Tensor
            tensor that needs sharding
        max_partition_sizes: List[int]
            for each number in the list, if world_size is smaller than or equal to that number, the tensor will get
            partitioned in worldsize parts, else if world size is larger than the number then you will get world size parts
            replicated in worldsize / number

            world_size = 4, max_partition_sizes = [8], tensor = [0 1 2 3 4 5 6 7]
            [0 1] [2 3] [4 5] [6 7]

            world_size = 8, max_partition_sizes = [4], tensor = [0 1 2 3 4 5 6 7]
            [0 1] [0 1] [2 3] [2 3] [4 5] [4 5] [6 7] [6 7]

            If there are multiple numbers in the max_partition_sizes list, then the param gets filled with non-contiguous
            slices of the tensor_value. This is useful for fused weight cases (qkv, mlp, moe, etc.)

            world_size = 4, max_partition_sizes = [4, 4], tensor = [0 1 2 3 4 5 6 7]
            [0 4] [1 5] [2 6] [3 7]

            world_size = 4, max_partition_sizes = [4, 1], tensor = [0 1 2 3 4 5 6 7 8 9]
            [0 1 8 9] [2 3 8 9] [4 5 8 9] [6 7 8 9]

        is_bias: bool
            if True is bias, else is weight
        """
        # Divide the weight matrix along the first dimension.
        cusum_max_partition_sizes = [0]
        min_partition_size = min(max_partition_sizes)
        for m in max_partition_sizes:
            cusum_max_partition_sizes.append(
                cusum_max_partition_sizes[-1] + (m // min_partition_size)
            )

        output_size_per_partition = param.shape[0] // (
            sum(max_partition_sizes) // min(max_partition_sizes)
        )
        if not is_bias:
            tensor = torch.cat(
                [
                    tensor_value[
                        self.__get_start_index(
                            tensor_value,
                            output_size_per_partition,
                            replication,
                            max_partition_sizes,
                            min_partition_size,
                            cusum_max_partition_sizes,
                            i,
                        ) : self.__get_end_index(
                            tensor_value,
                            output_size_per_partition,
                            replication,
                            max_partition_sizes,
                            min_partition_size,
                            cusum_max_partition_sizes,
                            i,
                        ),
                        :,
                    ]
                    for i, replication in enumerate(max_partition_sizes)
                ],
                dim=0,
            )
        else:
            tensor = torch.cat(
                [
                    tensor_value[
                        self.__get_start_index(
                            tensor_value,
                            output_size_per_partition,
                            replication,
                            max_partition_sizes,
                            min_partition_size,
                            cusum_max_partition_sizes,
                            i,
                        ) : self.__get_end_index(
                            tensor_value,
                            output_size_per_partition,
                            replication,
                            max_partition_sizes,
                            min_partition_size,
                            cusum_max_partition_sizes,
                            i,
                        )
                    ]
                    for i, replication in enumerate(max_partition_sizes)
                ],
                dim=0,
            )
        param.copy_(tensor, non_blocking=True)

    def load(
        self,
        param: torch.nn.Parameter,
        tensor_value: torch.Tensor,
        weight_name: str,
        is_bias: bool,
    ):
        """Load a tensor value into a param if it is marked as rowwise/colwise tensor parallel

        Args:
            param: torch.nn.Parameter
                the parameter to load the weight into
            tensor_value: torch.Tensor
                the tensor value to load into the parameter
            weight_name: str
                the weights name in the state_dict
            is_bias: bool
                True if param is referring to a bias, otherwise just a weight
        """
        if weight_name in self.colwise_params():
            self._copy_colwise(
                param, tensor_value, is_bias, self.colwise_params()[weight_name]
            )
        elif weight_name in self.rowwise_params():
            self._copy_rowwise(
                param, tensor_value, is_bias, self.rowwise_params()[weight_name]
            )

    @staticmethod
    @abstractmethod
    def import_module(module, group: ProcessGroup):
        pass

    def import_weights(self, module: nn.Module):
        for weight in self.colwise_param_names():
            apply_colwise_tp(
                getattr(self, weight),
                getattr(module, weight),
                self.world_size,
                self.rank,
            )
        for weight in self.rowwise_param_names():
            apply_rowwise_tp(
                getattr(self, weight),
                getattr(module, weight),
                self.world_size,
                self.rank,
            )
        for weight in self.embedding_param_names():
            apply_embedding_tp(
                getattr(self, weight),
                getattr(module, weight),
                self.world_size,
                self.rank,
            )
        tp_sharded_modules = list(
            itertools.chain(
                self.colwise_param_names(),
                self.rowwise_param_names(),
                self.embedding_param_names(),
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
