from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional
from importlib.util import find_spec

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
    """Parse linear configuration mapping to extract selected linear type from
    `linear_config['linear_type']`.
    `linear_type` can be string, callable, or None. Callable is a user-provided function
    to select linear type based on module name. It should return string or None.
    When no configuration is provided or linear type is None, we default to
    "torch_linear" type, which maps to torch.nn.Linear.
    """
    if not linear_config:
        return "torch_linear"

    linear_type = linear_config.get("linear_type", None)

    if not linear_type:
        return "torch_linear"

    linear_type_str = None
    if callable(linear_type):
        try:
            linear_type_from_callable = linear_type(module_name)
        except Exception as error:
            raise RuntimeError(
                "Error in user-provided function linear_type, while receiving "
                f"module_name={module_name if module_name is not None else 'None'}."
            ) from error
        if linear_type_from_callable is None:
            return "torch_linear"
        if not isinstance(linear_type_from_callable, str):
            raise TypeError(
                "Expected return from linear_type callable to be string but got "
                f"{type(linear_type_from_callable)} instead."
            )
        linear_type_str = linear_type_from_callable.lower()
        if linear_type_str not in __type_factory_map:
            raise ValueError(
                f"Unsupported linear_type `{linear_type_str}` returned by "
                "the callable set up in linear_config['linear_type']. Function failed "
                f"receiving module_name={module_name}. Check linear_type function."
            )
    if isinstance(linear_type, str):
        linear_type_str = linear_type.lower()
        if linear_type_str not in __type_factory_map:
            raise ValueError(
                f"Unsupported linear_type `{linear_type_str}` in linear_config."
            )
    if linear_type_str:
        return linear_type_str
    raise ValueError(
        "linear_type must be either a supported string or a module-selection function."
    )


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

    When the module quantization scheme depends on the module name,
    linear_config["linear_type"] will be a callable. In this case, `get_linear` first
    returns UninitializedLinear, such that a post-processing loop with access to all the
    module names can determine the correct module to instantiate.
    """
    if (
        linear_config
        and callable(linear_config.get("linear_type", None))
        and module_name is None
    ):
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


class FP8Linear(torch.nn.Module):
    """
    Class that handles FP8 weights loading and internally uses torchao to do the matmuls.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        linear_config: Mapping[str, Any],
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
        self.linear_config = linear_config

        assert self.linear_config["weights"] is not None, (
            "Weights must always be quantized for FP8Linear"
        )
        assert self.linear_config["weights"]["symmetric"], (
            "We only support symmetric weights for now"
        )
        assert not self.linear_config["weights"]["dynamic"], (
            "We only support pre-quantized weights for now"
        )

        # For torchao, tihs needs to call to_affine_quantized_floatx_static
        self.weight = torch.nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )

        weight_scale_shape = (
            (1,)
            if self.linear_config["weights"]["strategy"] == "tensor"
            else (in_features, 1)
        )
        self.weight_scale = torch.nn.Parameter(
            torch.ones(weight_scale_shape), requires_grad=False
        )

        self.has_bias = bias
        if self.has_bias:
            self.bias = torch.nn.Parameter(torch.zeros((out_features,)))

        # If this is true, then we need to call to_weight_tensor_with_linear_activation_quantization_metadata
        if (
            self.linear_config["input_activations"] is not None
            and not self.linear_config["input_activations"]["dynamic"]
        ):
            scale_shape = (
                (1,)
                if self.linear_config["input_activations"]["strategy"] == "tensor"
                else (in_features, 1)
            )
            self.input_scale = torch.nn.Parameter(
                torch.ones(scale_shape), requires_grad=False
            )

    def construct_torchao_structure(self) -> torch.Tensor:
        from torchao.quantization.granularity import PerTensor, PerRow
        from torchao.quantization.observer import get_block_size
        from torchao.quantization.quant_primitives import ZeroPointDomain
        from torchao.quantization.quant_api import _input_activation_quant_func_fp8
        from torchao.quantization.linear_activation_quantized_tensor import (
            to_linear_activation_quantized,
        )
        from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
        from torchao.dtypes.floatx.float8_layout import (
            Float8AQTTensorImpl,
            Float8Layout,
            Float8MMConfig,
        )
        from torchao.quantization.weight_tensor_linear_activation_quantization import (
            to_weight_tensor_with_linear_activation_quantization_metadata,
        )

        weight_granularity = (
            PerTensor()
            if self.linear_config["weights"]["strategy"] == "tensor"
            else PerRow()
        )
        fp8_layout = Float8Layout(Float8MMConfig(use_fast_accum=True))
        qweight = AffineQuantizedTensor(
            Float8AQTTensorImpl.from_plain(
                self.weight, self.weight_scale, None, fp8_layout
            ),
            get_block_size(self.weight.shape, weight_granularity),
            self.weight.shape,
            zero_point_domain=ZeroPointDomain.NONE,
            dtype=self.weight_scale.dtype,
        )

        if self.linear_config["input_activations"] is not None:
            input_quant_func = _input_activation_quant_func_fp8
            act_granularity = (
                PerTensor()
                if self.linear_config["input_activations"]["strategy"] == "tensor"
                else PerRow()
            )
            input_quant_kwargs = {
                "activation_granularity": act_granularity,
                "activation_dtype": torch.float8_e4m3fn,
            }

            if self.linear_config["input_activations"]["dynamic"]:
                qweight = to_linear_activation_quantized(
                    qweight, input_quant_func, input_quant_kwargs
                )
            else:
                qweight = to_weight_tensor_with_linear_activation_quantization_metadata(
                    qweight,
                    input_quant_func,
                    self.input_scale,
                    zero_point=None,
                    quant_kwargs=input_quant_kwargs,
                )

        return qweight

    def forward(self, x):
        qweight = self.construct_torchao_structure()
        return torch.nn.functional.linear(x, qweight)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(in={self.in_features}, out={self.out_features}, "
            f"bias={self.has_bias}, wq={self.weight_quant_type}, "
            f"aq={self.activ_quant_type}, smoothq={self.smoothquant}, "
            f"op={self.aiu_op})"
        )


def get_fp8_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config: Mapping[str, Any],
):
    if not find_spec("torchao"):
        raise ModuleNotFoundError("You need to install torchao for FP8 support in FMS!")

    return FP8Linear(in_features, out_features, bias, linear_config)


def shard_fp8_linear(
    tensor_values: dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: dict[str, LinearModuleShardingInfo],
) -> Optional[set]:
    """
                               |     GPU     |
    sharding  | param          | shard | dim |
    ----------+----------------+-------+-----|
    colwise   | weight         |   Y   |  0  |
              | weight_scale   |  Y/?  |  0  |
              | input_scale    |  Y/?  |  0  |
              | bias           |   Y   |  0  |
    ----------+----------------+-------+-----|
    rowwise   | weight         |   Y   |  1  |
              | weight_scale   |  Y/?  |  0  |
              | input_scale    |  Y/?  |  0  |
              | bias           |   0   |  -  |
    """
    param_sharding_info: dict[str, dict[str, LinearParameterShardingInfo]] = {}
    for module_name, module_info in module_sharding_info.items():
        linear_mod: torch.nn.Module = module_info.linear_module
        weight_strategy = getattr(linear_mod, "linear_config")["input_activations"][
            "strategy"
        ]
        shard_scales = weight_strategy != "tensor" and module_info.sharding_dim == 0
        params: dict[str, LinearParameterShardingInfo] = {
            "weight": LinearParameterShardingInfo(
                module_info.sharding_dim, ShardType.SHARD
            ),
            "weight_scale": LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if shard_scales else ShardType.CLONE,
            ),
        }
        if hasattr(linear_mod, "input_scale"):
            params["input_scale"] = LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if shard_scales else ShardType.CLONE,
            )
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


register_linear_type_to_module_map("fp8", get_fp8_linear)
register_linear_type_to_sharding_map("fp8", shard_fp8_linear)
