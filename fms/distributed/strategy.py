import os
from abc import abstractmethod
from typing import List

import torch
import torch.distributed
from torch import nn

from fms.utils import tp_wrapping


if "DISTRIBUTED_STRATEGY_IGNORE_MODULES" in os.environ:
    _distributed_strategy_ignore_modules = os.environ[
        "DISTRIBUTED_STRATEGY_IGNORE_MODULES"
    ].split(",")
else:
    _distributed_strategy_ignore_modules = []


class DistributedStrategy:
    def __init__(self, from_meta=False):
        self.from_meta = from_meta

    def __should_distribute(self, module_name: str) -> bool:
        return module_name not in _distributed_strategy_ignore_modules

    def distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        """
        Optionally a distributed strategy may distribute modules that are not
        numbered layers
        """
        module_name = type(module).__name__
        if self.__should_distribute(module_name):
            return self._distribute_module(module, final_layers)
        else:
            print(f"ignoring module={module_name} when distributing module")
            return module

    def distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        """
        Distribute each layer as-appropriate
        """
        block_name = type(block).__name__
        if self.__should_distribute(block_name):
            return self._distribute_layer(block, layer)
        else:
            print(f"ignoring block={block_name} when distributing layer")
            return block

    @abstractmethod
    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        """
        Distribute modules that are not numbered layers
        """
        pass

    @abstractmethod
    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        """
        Distribute each layer
        """
        pass


class NotDistributed(DistributedStrategy):
    def __init__(self, from_meta=False):
        super().__init__(from_meta)

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        return module

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        return block


NoOpStrategy = NotDistributed()


class DeviceMover(nn.Module):
    def __init__(self, module: nn.Module, device):
        super().__init__()
        self.device = device
        # make this wrapper module behave as if it was the wrapped module.
        attr = module.__dict__
        attr["module"] = module.to(device)
        attr["device"] = device
        self.__dict__ = attr

    def forward(self, *args, **kwargs):
        device = self.device
        args = [
            arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        kwargs = {
            k: (
                kwargs[k].to(device)
                if isinstance(kwargs[k], torch.Tensor)
                else kwargs[k]
            )
            for k in kwargs
        }
        return self.module(*args, **kwargs)


class UniformModelParallelStrategy(DistributedStrategy):
    def __init__(self, devices: List[int], num_layers: int, from_meta=False):
        super().__init__(from_meta)
        num_dev = len(devices)
        layers_per_dev = num_layers // num_dev
        remainder = num_layers - (layers_per_dev * num_dev)
        self.layer_to_device = [0] * num_layers
        layer_id = 0
        for dev_idx in range(len(devices)):
            for i in range(layers_per_dev):
                self.layer_to_device[layer_id] = devices[dev_idx]
                layer_id = layer_id + 1
            if remainder > 0:
                self.layer_to_device[layer_id] = devices[dev_idx]
                layer_id = layer_id + 1
                remainder -= 1

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        device = self.layer_to_device[layer]
        if self.from_meta:
            # https://github.com/pytorch/pytorch/pull/113647
            block.to_empty(device=device)  # type: ignore[arg-type]
        wrapped = DeviceMover(block, device)
        return wrapped

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        if final_layers:
            device = self.layer_to_device[len(self.layer_to_device) - 1]
        else:
            device = self.layer_to_device[0]
        if self.from_meta:
            return module.to_empty(device=device)  # type: ignore[arg-type]
        wrapped = DeviceMover(module, device)
        return wrapped


class TensorParallelStrategy(DistributedStrategy):
    def __init__(self, group=None, from_meta=False):
        super().__init__(from_meta)
        assert torch.distributed.is_initialized(), "must initialize a process group"
        self.group = group if group is not None else torch.distributed.GroupMember.WORLD

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        return tp_wrapping.apply_tp(module, self.group)

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        return tp_wrapping.apply_tp(block, self.group)



from torch import Tensor 
import torch.distributed as dist


class RingAttentionStrategy(DistributedStrategy):
    """
    Distributed strategy for ring attention with automatic input padding.
    Ensures tensors gathered across ranks are the same shape.
    """

    def __init__(self, block_size: int, group=None, from_meta=False):
        super().__init__(from_meta)
        assert torch.distributed.is_initialized(), "Requires initialized process group"
        self.group = group or torch.distributed.GroupMember.WORLD
        self.rank = self.group.rank()
        self.world_size = self.group.size()
        self.block_size = block_size
        self._original_seq_len = None

    def _distribute_module(self, module: nn.Module, final_layers: bool = False) -> nn.Module:
        return module

    def _distribute_layer(self, block: nn.Module, layer: int) -> nn.Module:
        return block

    def _pad_to_block_size(self, x: Tensor, dim: int = 1) -> Tensor:
        """Pads tensor along a dimension to block_size if needed."""
        length = x.size(dim)
        if length >= self.block_size:
            return x
        pad_shape = list(x.shape)
        pad_shape[dim] = self.block_size - length
        pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=dim)

    def shard_input(self, x: Tensor) -> Tensor:
        if self.world_size == 1:
            self._original_seq_len = x.size(1)
            self._local_valid_len = x.size(1)
            return x

        batch_size, seq_len = x.size(0), x.size(1)
        self._original_seq_len = seq_len
        start = self.rank * self.block_size
        end = min(start + self.block_size, seq_len)

        self._local_valid_len = end - start  # 👈 Fix is here
        shard = x[:, start:end, :]
        return self._pad_to_block_size(shard, dim=1)


    def gather_output(self, x_local: Tensor) -> Tensor:
        if self.world_size == 1:
            return x_local

        padded_list = [torch.empty_like(x_local) for _ in range(self.world_size)]
        dist.all_gather(padded_list, x_local.contiguous(), group=self.group)
        full = torch.cat(padded_list, dim=1)
        if self._original_seq_len and full.size(1) > self._original_seq_len:
            return full[:, :self._original_seq_len]
        return full

    def gather_tensor(self, tensor: Tensor, dim: int = 1) -> Tensor:
        if self.world_size == 1:
            return tensor

        tensor = self._pad_to_block_size(tensor, dim=dim)
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=self.group)
        result = torch.cat(gathered, dim=dim)

        if dim == 1 and self._original_seq_len and result.size(dim) > self._original_seq_len:
            result = result.narrow(dim, 0, self._original_seq_len)
        return result
