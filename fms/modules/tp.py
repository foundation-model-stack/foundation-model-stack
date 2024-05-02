import itertools
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Set, Type, Union

import torch
import torch.nn as nn
from torch.distributed.distributed_c10d import ProcessGroup


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

    def __get_tp_slices(
        self,
        input_size,
        output_size_per_partition,
        max_partition_sizes,
    ):
        cusum_max_partition_sizes = [0]
        min_partition_size = min(max_partition_sizes)
        for m in max_partition_sizes:
            cusum_max_partition_sizes.append(
                cusum_max_partition_sizes[-1] + (m // min_partition_size)
            )

        for weight_i, replication in enumerate(max_partition_sizes):
            # The shard to copy based on the process rank and the replication threshold
            sharding_rank = self.rank // max(1, self.world_size // replication)
            # The number of elements to shard out of the tensor
            tensor_shard_size = output_size_per_partition * (
                max_partition_sizes[weight_i] // min_partition_size
            )
            # For fused weights, where to start extracting the shard
            tensor_shard_offset = (
                cusum_max_partition_sizes[weight_i]
                * input_size
                // cusum_max_partition_sizes[-1]
            )
            yield slice(
                sharding_rank * tensor_shard_size + tensor_shard_offset,
                (sharding_rank + 1) * tensor_shard_size + tensor_shard_offset,
            )

    def copy_rowwise(
        self,
        param: Union[torch.nn.Parameter, torch.Tensor],
        tensor_value: torch.Tensor,
        max_partition_sizes: List[int],
        is_sharded: bool = True,
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
        is_sharded: bool
            For additive terms (like bias), is_sharded must be False. Otherwise True.
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
        """
        if is_sharded:
            # Divide the weight matrix along the second dimension.
            output_size_per_partition = param.shape[1] // (
                sum(max_partition_sizes) // min(max_partition_sizes)
            )
            tp_slices = self.__get_tp_slices(
                tensor_value.shape[1], output_size_per_partition, max_partition_sizes
            )
            tensors_to_cat = [tensor_value[:, tp_slice] for tp_slice in tp_slices]
            tensor = torch.cat(tensors_to_cat, dim=1)
            param.copy_(tensor, non_blocking=True)
        else:
            if self.rank == 0:
                param.copy_(tensor_value, non_blocking=True)
            else:
                param.zero_()

    def copy_colwise(
        self,
        param: Union[torch.nn.Parameter, torch.Tensor],
        tensor_value: torch.Tensor,
        max_partition_sizes: List[int],
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
        """
        # Divide the weight matrix along the first dimension.
        output_size_per_partition = param.shape[0] // (
            sum(max_partition_sizes) // min(max_partition_sizes)
        )
        tp_slices = self.__get_tp_slices(
            tensor_value.shape[0], output_size_per_partition, max_partition_sizes
        )
        tensors_to_cat = [tensor_value[tp_slice,] for tp_slice in tp_slices]
        tensor = torch.cat(tensors_to_cat, dim=0)
        param.copy_(tensor, non_blocking=True)

    def _get_sd_weight(
        self,
        state_dict: Dict[str, torch.Tensor],
        used_keys: Set[str],
        substr_matches: List[str],
    ):
        results = []
        for k in state_dict:
            all_matches = True
            for substr_match in substr_matches:
                if substr_match not in k:
                    all_matches = False
            if all_matches:
                used_keys.add(k)
                results.append(state_dict[k])
        if len(results) == 1:
            return results[0]
        elif len(results) > 1:
            raise ValueError(
                f"Conditions not stringent enough: keys are {', '.join(state_dict.keys())}"
            )
        else:
            raise ValueError(
                f"Weight not found, weights names are {', '.join(state_dict.keys())}"
            )

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        """Load all tensor values into a TP module.

        Override this method to load weights into a TP module. You can see
        examples for all TP modules in FMS, but the functions will generally
        have the following structure:

        1. Grab the necessary weights from tensor_values. Which weights
            these are depend on each module.
        2. Raise exceptions for missing required weights in tensor_values
            (ValueError) or for unused weights in tensor_values
            (AttributeError)
        3. Use one of the copy_{rowwise,colwise} methods to load and shard
            each weight correctly into the TP module. All the copy_ methods
            explain in detail how to call them properly. Pay special attention
            to the max_partition_sizes parameter for supporting fused weights
            and MQA/GQA among others.

        PSA: Each weight has a List[int] (max_partition_sizes) associated
        with it where if the list is larger than size 1, this implies the
        weight is fused (where the length of the list is the number of fused
        weights in a single parameter).

        Args:
            tensor_values: Dict[str, torch.Tensor]
                a state dict containing all the weights for a TP module
        """
        pass

    @staticmethod
    @abstractmethod
    def import_module(module, group: ProcessGroup):
        pass
