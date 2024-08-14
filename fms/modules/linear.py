import torch
import torch.nn as nn
from typing import Callable, Mapping, Any


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
    tensor_values: torch.Tensor,
    tp_module,  # hint should be: type(TPMultiHeadAttention) | type(TPFeedForwardBlock)
    modules: list[str],
    params: list[str],
    name_to_module: dict[str, nn.Module],
    module_base_shard_dim: dict[str, int],
    max_partition: dict[str, str],
    unsharded: list[tuple[str, str]],
) -> None:
    all_params = {}
    used_keys: set[str] = set()

    # Collect quantization parameters to copy on sharded module
    tensor_device = None
    for module_name in modules:
        for param_name in params:
            if module_name not in all_params:
                all_params[module_name] = {}
            # TODO: reusing method '_get_sd_weight' but consider changing its name
            all_params[module_name][param_name] = tp_module._get_sd_weight(
                tensor_values, used_keys, [module_name, param_name]
            )
            if tensor_device is None:
                tensor_device = all_params[module_name][param_name].device

    # TODO: fix used_keys validation
    # if len(tensor_values) > (8 if self.use_bias else 4):
    #     unused_keys = set(tensor_values.keys()).difference(used_keys)
    #     raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

    # Shard module, one parameter at the time
    for module_name in modules:
        for param_name in params:
            module_param = getattr(name_to_module[module_name], param_name)
            if module_param.device == torch.device("meta"):
                if isinstance(module_param, nn.Parameter):
                    module_param = nn.Parameter(torch.empty_like(module_param, device=tensor_device))
                else:
                    module_param = torch.empty_like(module_param, device=tensor_device)
                setattr(
                    name_to_module[module_name],
                    param_name,
                    module_param
                )
                module_param = getattr(name_to_module[module_name], param_name)

            is_sharded = not any([m == module_name and p == param_name for (m, p) in unsharded])

            shard_dim_param = module_base_shard_dim[module_name]
            # TODO: need to bring this into shard_torch_linear and shard_gptq_linear... HOW?
            if param_name in ["bias", "g_idx"]:
                shard_dim_param = 0

            tp_module.sharded_copy(
                param=module_param,
                tensor_value=all_params[module_name][param_name],
                dim=shard_dim_param,
                max_partition_sizes=max_partition[module_name],
                is_sharded=is_sharded,
            )


def shard_torch_linear(
    tensor_values: torch.Tensor,
    tp_module,  # hint should be: type(TPMultiHeadAttention) | type(TPFeedForwardBlock)
    modules: list[str],
    name_to_module: dict[str, nn.Module],
    module_base_shard_dim: dict[str, int],
    max_partition: dict [str, str],
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
    params = ["weight"]
    if tp_module.use_bias:
        params.append("bias")

    # List of tuples (module, param) that won't be sharded
    if "dense" in modules:
        unsharded = [("dense", "bias")]
    elif "w2" in modules:
        unsharded = [("w2", "bias")]

    shard_base_linear(
        tensor_values,
        tp_module,
        modules,
        params,
        name_to_module,
        module_base_shard_dim,
        max_partition,
        unsharded,
    )


register_linear_type_to_module_map("torch_linear", nn.Linear)
register_linear_type_to_sharding_map("torch_linear", shard_torch_linear)
