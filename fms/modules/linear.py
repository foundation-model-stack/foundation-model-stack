import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import torch
from torch import nn

from fms.modules import UninitializedModule
from fms.modules.tp import ShardType, TPModule


__type_factory_map: dict[str, Callable] = {}
__type_sharding_map: dict[str, Callable] = {}


def register_linear_type_to_module_map(linear_type: str, factory: Callable) -> None:
    """Registration of a linear type (e.g., "gptq") and associated module / module
    factory function.
    Registered module will be made available at the time a model is built, to be
    instantiated by `get_linear`.
    This function can be called from other scripts to register custom modules.
    """
    if linear_type in __type_factory_map:
        raise KeyError(
            f"Module mapping of linear type `{linear_type}` already registered"
        )
    __type_factory_map[linear_type] = factory


def register_linear_type_to_sharding_map(linear_type: str, factory: Callable) -> None:
    """Registration of a linear type (e.g., "gptq") and associated Tensor Parallel (TP)
    sharding function (e.g., `shard_gptq_linear`).
    The sharding function determines how the parameters of a module are to be sharded
    with TP.
    This function can be called from other scripts to register custom TP sharding
    functionalities.
    """
    if linear_type in __type_sharding_map:
        raise KeyError(
            f"Sharding map of linear type `{linear_type}` already registered"
        )
    __type_sharding_map[linear_type] = factory


def get_all_linear_type_to_sharding_maps() -> dict[str, Callable]:
    """Return all currently registered mappings from linear types to TP sharding
    functions.
    """
    return __type_sharding_map


def get_linear_type(
    linear_config: Optional[Mapping[str, Any]], module_name: Optional[str] = None
) -> str:
    """Parse linear configuration mapping to extract selected linear type.
    If no mapping is provided, defaults to "torch_linear" type, which maps
    to torch.nn.Linear.
    """
    if not linear_config:
        return "torch_linear"

    if (
        "filter_fn" in linear_config
        and module_name is not None
        and not linear_config["filter_fn"](module_name)
    ):
        return "torch_linear"

    linear_type = linear_config.get("linear_type", None)
    if not linear_type:
        return "torch_linear"
    if not isinstance(linear_type, str):
        raise TypeError("linear_type in linear_config must be string")
    if linear_type.lower() not in __type_factory_map:
        raise ValueError(f"Unsupported linear_type `{linear_type}` in linear_config")

    return linear_type.lower()


class UninitializedLinear(UninitializedModule):
    def __init__(self, in_features, out_features, bias, linear_config):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear_config = linear_config

    def initialize(self, name):
        return get_linear(
            self.in_features, self.out_features, self.bias, self.linear_config, name
        )


def get_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config: Optional[Mapping[str, Any]] = None,
    module_name: Optional[str] = None,
) -> nn.Module:
    """Return linear module or module factory function of selected type.
    Linear type is extracted from provided configuration (`linear_config`) and
    associated module is determined from existing mapping (`__type_factory_map`).
    Selected module must have been registered with `register_linear_type_to_module_map`.

    In the cases where a module can be quantized or not depending on its name (given by
    boolean check in filter_fn as part of linear_config, True meaning quantize), returns
    UnitializedLinear so a later loop can correctly initialize the model.
    """
    if linear_config and "filter_fn" in linear_config and module_name is None:
        return UninitializedLinear(in_features, out_features, bias, linear_config)

    linear_type = get_linear_type(linear_config, module_name)

    if linear_type in __type_factory_map:
        if linear_type == "torch_linear":
            return __type_factory_map[linear_type](in_features, out_features, bias)
        return __type_factory_map[linear_type](
            in_features, out_features, bias, linear_config
        )
    raise KeyError(f"Unsupported linear type `{linear_type}`")


@dataclass
class LinearModuleShardingInfo:
    linear_module: torch.nn.Module
    sharding_dim: int
    max_partitions: list[int]


@dataclass
class LinearParameterShardingInfo:
    sharding_dim: int
    shard_type: ShardType


def shard_base_linear(
    tensor_values: dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: dict[str, LinearModuleShardingInfo],
    param_sharding_info: dict[str, dict[str, LinearParameterShardingInfo]],
) -> Optional[set]:
    """Base Tensor Parallel (TP) sharding function for linear layers.
    Using a dictionary of parameter names and unsharded tensors (`tensor_values`),
    and a TP-enabled module (`tp_module`), this function copies the correct shard
    from each tensor into the corresponding sharded module parameter.
    """
    all_params: dict = {}
    used_keys: set[str] = set()
    unused_keys: set[str] = set()

    # Collect all parameters to be copied on selected sharded modules
    param_count = 0
    for module_name in module_sharding_info:
        for param_name in param_sharding_info[module_name]:
            if module_name not in all_params:
                all_params[module_name] = {}
            all_params[module_name][param_name] = tp_module._get_sd_weight(
                tensor_values, used_keys, [module_name, param_name]
            )
            param_count += 1

    if len(tensor_values) > param_count:
        unused_keys = set(tensor_values.keys()).difference(used_keys)

    # Shard selected modules, one parameter at the time
    for module_name, module_info in module_sharding_info.items():
        for param_name, param_info in param_sharding_info[module_name].items():
            module_param = getattr(module_info.linear_module, param_name)
            tp_module.sharded_copy(
                param=module_param,
                tensor_value=all_params[module_name][param_name],
                dim=param_info.sharding_dim,
                max_partition_sizes=module_info.max_partitions,
                shard_type=param_info.shard_type,
            )
    return unused_keys


def shard_torch_linear(
    tensor_values: dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: dict[str, LinearModuleShardingInfo],
) -> Optional[set]:
    """
                         |     GPU     |
    sharding  | param    | shard | dim |
    ----------+----------+-------+-----|
    colwise   | weight   |   Y   |  0  |
              | bias     |   Y   |  0  |
    ----------+----------+-------+-----|
    rowwise   | weight   |   Y   |  1  |
              | bias     |   0   |  -  |
    """
    param_sharding_info: dict[str, dict[str, LinearParameterShardingInfo]] = {}
    for module_name, module_info in module_sharding_info.items():
        linear_mod: torch.nn.Module = module_info.linear_module
        params: dict[str, LinearParameterShardingInfo] = {
            "weight": LinearParameterShardingInfo(
                module_info.sharding_dim, ShardType.SHARD
            )
        }
        if hasattr(linear_mod, "bias") and linear_mod.bias is not None:
            params["bias"] = LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.RANK0,
            )
        param_sharding_info[module_name] = params

    unused_keys = shard_base_linear(
        tensor_values,
        tp_module,
        module_sharding_info,
        param_sharding_info,
    )
    return unused_keys


register_linear_type_to_module_map("torch_linear", nn.Linear)
register_linear_type_to_sharding_map("torch_linear", shard_torch_linear)
