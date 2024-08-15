import torch
import torch.nn as nn
from typing import Callable, Dict, List, Mapping, Any, Tuple
from fms.modules.tp import TPModule


__type_factory_map: Mapping[str, Callable] = {}
__type_sharding_map : Mapping[str, Callable] = {}


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


def get_linear_type(linear_config: Mapping[str, Any] | None) -> str:
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
    linear_config: Mapping[str, Any] | None = None,
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


def shard_base_linear(
    tensor_values: dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: Dict[str, Tuple[torch.nn.Module, List[int], List[Tuple[str, int, bool]]]],
) -> None:
    all_params = {}
    used_keys: set[str] = set()

    # Collect quantization parameters to copy on sharded module
    tensor_device = None
    param_count = 0
    for module_name, module_info in module_sharding_info.items():
        for param_name, _, _ in module_info[2]:
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
         for param_name, param_shard_dim, param_sharding in module_info[2]:
            module_param = getattr(module_info[0], param_name)
            if module_param.device == torch.device("meta"):
                # TODO: We should bring the default dtype if set from get_model here
                if isinstance(module_param, nn.Parameter):
                    module_param = nn.Parameter(torch.empty_like(module_param, device=tensor_device))
                else:
                    module_param = torch.empty_like(module_param, device=tensor_device)
                setattr(
                    module_info[0],
                    param_name,
                    module_param
                )
                module_param = getattr(module_info[0], param_name)

            # TODO: need to bring this into shard_torch_linear and shard_gptq_linear... HOW? -> Use new struct
            # if param_name in ["bias", "g_idx"]:
            #     shard_dim_param = 0

            tp_module.sharded_copy(
                param=module_param,
                tensor_value=all_params[module_name][param_name],
                dim=param_shard_dim,
                max_partition_sizes=module_info[1],
                is_sharded=param_sharding,
            )


def shard_torch_linear(
    tensor_values: Dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: Dict[str, Tuple[torch.nn.Module, List[int], List[Tuple[str, int, bool]]]],
) -> None:
    """
                         |     GPU     |
    module    | param    | shard | dim |
    ----------+----------+-------+-----|
    QKV, w1   | weight   |   Y   |  0  |
              | bias     |   Y   |  0  |
    ----------+----------+-------+-----|
    dense, w2 | weight   |   Y   |  1  |
              | bias     |   N   |  -  |
    """
    shard_base_linear(
        tensor_values,
        tp_module,
        module_sharding_info,
    )


register_linear_type_to_module_map("torch_linear", nn.Linear)
register_linear_type_to_sharding_map("torch_linear", shard_torch_linear)
