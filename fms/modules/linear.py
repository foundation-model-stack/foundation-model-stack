import torch.nn as nn
from fms.utils.gptq import get_gptq_linear
from typing import Mapping, Any


# TODO: selection of Linear needs to be updatable from external calls
# how to register additional mappings into this?
TYPE_FACTORY_MAP = {}
TYPE_FACTORY_MAP["torch_linear"] = nn.Linear
TYPE_FACTORY_MAP["gptq"] = get_gptq_linear
# TYPE_FACTORY_MAP["fp8"]


def _get_linear_type(linear_config: Mapping[str, Any] | None) -> str:
    if not linear_config:
        return "torch_linear"
    linear_type = linear_config.get("linear_type", None)
    if not linear_type:
        return "torch_linear"
    if not isinstance(linear_type, str):
        raise TypeError("linear_type in linear_config must be string")
    if linear_type not in TYPE_FACTORY_MAP:
        raise ValueError(f"Unsupported linear_type in linear_config: `{linear_type}`")
    return linear_type.lower()


def get_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config: Mapping[str, Any] | None = None,
) -> nn.Module:
    linear_type = _get_linear_type(linear_config)

    # TODO: how to merge these calls that get different arguments?
    if linear_type in TYPE_FACTORY_MAP:
        if linear_type == "torch_linear":
            return TYPE_FACTORY_MAP[linear_type](in_features, out_features, bias)
        else:
            return TYPE_FACTORY_MAP[linear_type](
                in_features, out_features, bias, linear_config
            )
    else:
        raise TypeError(f"Unsupported linear type `{linear_type}`")
