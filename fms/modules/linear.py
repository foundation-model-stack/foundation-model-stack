import torch.nn as nn
from fms.utils.gptq import get_gptq_linear
from typing import Mapping, Any


# TODO: selection of Linear needs to be updatable from external calls
# how to register additional mappings into this?
LINEAR_MAPPING = {}
LINEAR_MAPPING["TORCH_LINEAR"] = nn.Linear
LINEAR_MAPPING["GPTQ"] = get_gptq_linear
# LINEAR_MAPPING["FP8"]


def _get_linear_type(linear_config: Mapping[str, Any] | None) -> str:
    if not linear_config:
        return "TORCH_LINEAR"
    linear_type = linear_config.get("linear_type", None)
    if not linear_type:
        return "TORCH_LINEAR"
    if not isinstance(linear_type, str):
        raise TypeError("linear_type in linear_config must be string")
    if linear_type not in LINEAR_MAPPING:
        raise ValueError(f"Unsupported linear_type in linear_config: `{linear_type}`")
    return linear_type.upper()


def get_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config: Mapping[str, Any] | None = None,
) -> nn.Module:
    linear_type = _get_linear_type(linear_config)

    # TODO: how to merge these calls that get different parameters?
    if linear_type == "TORCH_LINEAR":
        return LINEAR_MAPPING[linear_type](in_features, out_features, bias)
    else:
        return LINEAR_MAPPING[linear_type](
            in_features, out_features, bias, linear_config
        )