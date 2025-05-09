import os
from abc import abstractmethod
from typing import List, Optional, Tuple
from torch.distributed import P2POp

import torch
import torch.distributed
from torch import nn

from fms.utils import tp_wrapping

import torch
from torch import Tensor, nn
import torch.distributed as dist # Keep this for P2POp if not already imported
from typing import Optional


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

class RingAttentionStrategy(DistributedStrategy):
    def __init__(
        self,
        block_size: int = 2048,
        group: Optional[dist.ProcessGroup] = None,
        from_meta: bool = False
    ):
        super().__init__(from_meta)
        self.block_size = block_size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.group = group
            self.rank = torch.distributed.get_rank(group=self.group)
            self.world_size = torch.distributed.get_world_size(group=self.group)
        else:
            self.group = None
            self.rank = 0
            self.world_size = 1
            print(
                "[INFO] RingAttentionStrategy: torch.distributed not initialized,"
                " defaulting to world_size=1, rank=0."
            )
        self._original_seq_len: Optional[int] = None
        self._local_valid_len: Optional[int] = None

    def _pad_to_block_size(
        self, tensor: torch.Tensor, dim: int = 1
    ) -> torch.Tensor:
        length = tensor.size(dim)
        if length == self.block_size:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[dim] = self.block_size - length
        padding = torch.zeros(*pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=dim)

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False
    ) -> nn.Module:
        return module

    def _distribute_layer(
        self, block: nn.Module, layer: int
    ) -> nn.Module:
        return block

    def shard_input(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        seq_len = x.size(1)
        self._original_seq_len = seq_len

        if self.world_size == 1:
            self._local_valid_len = seq_len
            return x
        start = self.rank * self.block_size
        end = min(start + self.block_size, seq_len)
        self._local_valid_len = max(0, end - start)
        if self._local_valid_len > 0:
            return x.narrow(1, start, self._local_valid_len)
        shp = list(x.shape)
        shp[1] = 0
        return torch.empty(*shp, dtype=x.dtype, device=x.device)

    def _ring_shift_tensor(
        self,
        tensor: torch.Tensor,
        valid_seq_len: int
    ) -> Tuple[torch.Tensor, int]:
        if self.world_size == 1:
            if valid_seq_len == 0:
                empty_shape = list(tensor.shape)
                empty_shape[2] = 0
                return torch.empty(*empty_shape, dtype=tensor.dtype, device=tensor.device), 0
            idx = [slice(None)] * tensor.ndim
            idx[2] = slice(0, valid_seq_len)
            return tensor[tuple(idx)].clone(), valid_seq_len

        send_to = (self.rank + 1) % self.world_size
        recv_from = (self.rank - 1 + self.world_size) % self.world_size
        seq_dim = 2

        if valid_seq_len == 0:
            empty_shape = list(tensor.shape)
            empty_shape[seq_dim] = 0
            to_send = torch.empty(*empty_shape, dtype=tensor.dtype, device=tensor.device)
        else:
            idx = [slice(None)] * tensor.ndim
            idx[seq_dim] = slice(0, valid_seq_len)
            to_send = tensor[tuple(idx)]

        padded = self._pad_to_block_size(to_send, dim=seq_dim).contiguous()
        recv_buf = torch.empty_like(padded)
        recv_len = torch.empty(1, dtype=torch.int32, device=tensor.device)
        send_len = torch.tensor([valid_seq_len], dtype=torch.int32, device=tensor.device)

        ops = [
            P2POp(dist.isend, send_len, peer=send_to),
            P2POp(dist.irecv, recv_len, peer=recv_from),
            P2POp(dist.isend, padded, peer=send_to),
            P2POp(dist.irecv, recv_buf, peer=recv_from)
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        new_len = recv_len.item()
        assert 0 <= new_len <= self.block_size
        idx2 = [slice(None)] * recv_buf.ndim
        idx2[seq_dim] = slice(0, new_len)
        return recv_buf[tuple(idx2)].contiguous(), new_len

    def get_local_valid_len(self) -> int:
        assert self._local_valid_len is not None
        return self._local_valid_len

    def gather_tensor(
        self, tensor: torch.Tensor, dim: int = 1
    ) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        t = tensor.contiguous()
        if t.size(dim) != self.block_size:
            t = self._pad_to_block_size(t, dim)
        gathered = [torch.empty_like(t) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, t, group=self.group)
        result = torch.cat(gathered, dim=dim)
        if dim == 1:
            assert self._original_seq_len is not None
            result = result.narrow(dim, 0, self._original_seq_len)
        return result
