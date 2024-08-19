from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

from fms.modules.tp import ShardType, TPModule


__type_factory_map: Mapping[str, Callable] = {}
__type_sharding_map: Mapping[str, Callable] = {}


def register_linear_type_to_module_map(linear_type: str, factory: Callable) -> None:
    if linear_type in __type_factory_map:
        raise KeyError(
            f"Module mapping of linear type `{linear_type}` already registered"
        )
    __type_factory_map[linear_type] = factory


def register_linear_type_to_sharding_map(linear_type: str, factory: Callable) -> None:
    if linear_type in __type_sharding_map:
        raise KeyError(
            f"Sharding map of linear type `{linear_type}` already registered"
        )
    __type_sharding_map[linear_type] = factory


def get_all_linear_type_to_sharding_maps() -> dict[str, Callable]:
    return __type_sharding_map


def get_linear_type(linear_config:  Optional[Mapping[str, Any]]) -> str:
    if not linear_config:
        return "torch_linear"

    linear_type = linear_config.get("linear_type", None)
    if not linear_type:
        return "torch_linear"
    if not isinstance(linear_type, str):
        raise TypeError("linear_type in linear_config must be string")
    if linear_type.lower() not in __type_factory_map:
        raise ValueError(f"Unsupported linear_type `{linear_type}` in linear_config")

    return linear_type.lower()


def get_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config:  Optional[Mapping[str, Any]] = None,
) -> nn.Module:
    linear_type = get_linear_type(linear_config)

    # TODO: how to merge these calls that get different arguments?
    if linear_type in __type_factory_map:
        if linear_type == "torch_linear":
            return __type_factory_map[linear_type](in_features, out_features, bias)
        else:
            return __type_factory_map[linear_type](
                in_features, out_features, bias, linear_config
            )
    else:
        raise KeyError(f"Unsupported linear type `{linear_type}`")


@dataclass
class LinearModuleShardingInfo:
    linear_module: torch.nn.Module
    sharding_dim: int
    max_partitions: List[int]


@dataclass
class LinearParameterShardingInfo:
    sharding_dim: int
    shard_type: ShardType


def shard_base_linear(
    tensor_values: Dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: Dict[str, LinearModuleShardingInfo],
    param_sharding_info: Dict[str, Dict[str, LinearParameterShardingInfo]],
) -> None:
    all_params = {}
    used_keys: set[str] = set()

    # Collect quantization parameters to copy on sharded module
    tensor_device = None
    param_count = 0
    for module_name in module_sharding_info:
        for param_name in param_sharding_info[module_name]:
            if module_name not in all_params:
                all_params[module_name] = {}
            # TODO: reusing method '_get_sd_weight' but consider changing its name
            all_params[module_name][param_name] = tp_module._get_sd_weight(
                tensor_values, used_keys, [module_name, param_name]
            )
            # TODO: We should transfer the initial_device here from get_model
            if tensor_device is None:
                tensor_device = all_params[module_name][param_name].device
            param_count += 1

    if len(tensor_values) > param_count:
        unused_keys = set(tensor_values.keys()).difference(used_keys)
        raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

    # Shard module, one parameter at the time
    for module_name, module_info in module_sharding_info.items():
        for param_name, param_info in param_sharding_info[module_name].items():
            module_param = getattr(module_info.linear_module, param_name)
            if module_param.device == torch.device("meta"):
                # TODO: We should bring the default dtype if set from get_model here
                if isinstance(module_param, nn.Parameter):
                    module_param = nn.Parameter(
                        torch.empty_like(module_param, device=tensor_device)
                    )
                else:
                    module_param = torch.empty_like(module_param, device=tensor_device)
                setattr(module_info.linear_module, param_name, module_param)
                module_param = getattr(module_info.linear_module, param_name)

            tp_module.sharded_copy(
                param=module_param,
                tensor_value=all_params[module_name][param_name],
                dim=param_info.sharding_dim,
                max_partition_sizes=module_info.max_partitions,
                shard_type=param_info.shard_type,
            )


def shard_torch_linear(
    tensor_values: Dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: Dict[str, LinearModuleShardingInfo],
) -> None:
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
    param_sharding_info: Dict[str, Dict[str, LinearParameterShardingInfo]] = {}
    for module_name, module_info in module_sharding_info.items():
        linear_mod: torch.nn.Linear = module_info.linear_module
        params: Dict[str, LinearParameterShardingInfo] = {
            "weight": LinearParameterShardingInfo(
                module_info.sharding_dim, ShardType.SHARD
            )
        }
        if linear_mod.bias is not None:
            params["bias"] = LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.RANK0,
            )
        param_sharding_info[module_name] = params

    shard_base_linear(
        tensor_values,
        tp_module,
        module_sharding_info,
        param_sharding_info,
    )


register_linear_type_to_module_map("torch_linear", nn.Linear)
register_linear_type_to_sharding_map("torch_linear", shard_torch_linear)
