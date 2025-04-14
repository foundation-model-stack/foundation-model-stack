import os
from abc import abstractmethod
from typing import List

import torch
import torch.distributed
from torch import nn

from fms.utils import tp_wrapping
from torch.distributed.device_mesh import init_device_mesh


from torch.distributed.tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
   parallelize_module,
   ColwiseParallel,
   RowwiseParallel,
   SequenceParallel,
   PrepareModuleInput,
   PrepareModuleOutput
)


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
        self, module: nn.Module, final_layers: bool = False, model = None
    ) -> nn.Module:
        """
        Optionally a distributed strategy may distribute modules that are not
        numbered layers
        """
        module_name = type(module).__name__
        if self.__should_distribute(module_name):
            return self._distribute_module(module, final_layers, model)
        else:
            print(f"ignoring module={module_name} when distributing module")
            return module

    def distribute_layer(self, block: nn.Module, layer: int, model = None) -> nn.Module:
        """
        Distribute each layer as-appropriate
        """
        block_name = type(block).__name__
        if self.__should_distribute(block_name):
            return self._distribute_layer(block, layer, model)
        else:
            print(f"ignoring block={block_name} when distributing layer")
            return block

    @abstractmethod
    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False, model = None
    ) -> nn.Module:
        """
        Distribute modules that are not numbered layers
        """
        pass

    @abstractmethod
    def _distribute_layer(self, block: nn.Module, layer: int, model = None) -> nn.Module:
        """
        Distribute each layer
        """
        pass


class NotDistributed(DistributedStrategy):
    def __init__(self, from_meta=False):
        super().__init__(from_meta)

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False, model = None
    ) -> nn.Module:
        return module

    def _distribute_layer(self, block: nn.Module, layer: int, model = None) -> nn.Module:
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

    def _distribute_layer(self, block: nn.Module, layer: int, model = None) -> nn.Module:
        device = self.layer_to_device[layer]
        if self.from_meta:
            # https://github.com/pytorch/pytorch/pull/113647
            block.to_empty(device=device)  # type: ignore[arg-type]
        wrapped = DeviceMover(block, device)
        return wrapped

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False, model = None
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
        self.use_sequence_parallelism = os.getenv("USE_SEQUENCE_PARALLELISM", False)
        if self.use_sequence_parallelism:
            print("Using TP strategy with sequence parallelism")
        else:
            print("Using TP strategy without sequence parallelism")
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        world = torch.distributed.get_world_size()
        self.device_mesh = init_device_mesh(device_type, (world,))

    def _distribute_module(
        self, module: nn.Module, final_layers: bool = False, model = None
    ) -> nn.Module:
        if not model:
            return tp_wrapping.apply_tp(module, self.group)
        elif model == 'llama': 
            if final_layers:
                tp_plan = {
                    "shared.head": ColwiseParallel(output_layouts=Replicate(),),
                }
                return parallelize_module(module, self.device_mesh, tp_plan)
            else:
                tp_plan = {
                    "shared.emb": RowwiseParallel(input_layouts=Replicate()),
                }
                return parallelize_module(module, self.device_mesh, tp_plan)
        elif model == 'granite':
            tp_plan = {
                "head": ColwiseParallel(output_layouts=Replicate(),),
                "base_model.embedding": RowwiseParallel(input_layouts=Replicate()),
            }
            return parallelize_module(module, self.device_mesh, tp_plan)

    def _distribute_layer(self, block: nn.Module, layer: int, model = None) -> nn.Module:
        if not model:
            return tp_wrapping.apply_tp(block, self.group)
        elif model == 'llama':
            layer_tp_plan = {
                "attn.in_proj.qkv_fused": ColwiseParallel(),
                "attn.in_proj.query": ColwiseParallel(),
                "attn.in_proj.key": ColwiseParallel(),
                "attn.in_proj.value": ColwiseParallel(),
                "attn.dense": RowwiseParallel(),
                "ff_sub_layer.wg": ColwiseParallel(),
                "ff_sub_layer.wg1_fused": ColwiseParallel(),
                "ff_sub_layer.w2": RowwiseParallel(),
                "ff_sub_layer.w1": ColwiseParallel(),
                }
        elif model == 'granite':
            layer_tp_plan = {
                "attn.in_proj.qkv_fused": ColwiseParallel(),
                "attn.in_proj.query": ColwiseParallel(),
                "attn.in_proj.key": ColwiseParallel(),
                "attn.in_proj.value": ColwiseParallel(),
                "attn.dense": RowwiseParallel(),
                "ff_sub_layer.wg": ColwiseParallel(),
                "ff_sub_layer.wg1_fused": ColwiseParallel(),
                "ff_sub_layer.w2": RowwiseParallel(),
                "ff_sub_layer.w1": ColwiseParallel(),
                }
        else:
            raise ValueError(f"Unsupported model: {model}")
        # Adjust attention module to use the local number of heads
        attn_layer = block.attn
        attn_layer.nheads = attn_layer.nheads // self.device_mesh.size()
        attn_layer.kvheads = attn_layer.kvheads // self.device_mesh.size()

        #Custom parallelization plan for the model
        return parallelize_module(
            module=block,
            device_mesh=self.device_mesh,
            parallelize_plan=layer_tp_plan
        )
