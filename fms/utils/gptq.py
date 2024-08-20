from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn

from fms.modules.linear import (
    LinearModuleShardingInfo,
    LinearParameterShardingInfo,
    register_linear_type_to_module_map,
    register_linear_type_to_sharding_map,
    shard_base_linear,
)
from fms.modules.tp import ShardType, TPModule
from fms.utils.config import ModelConfig


try:
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

    IS_AUTOGPTQ_AVAILABLE = True
except:
    IS_AUTOGPTQ_AVAILABLE = False


# simplified from AutoGPTQ quantization config
# see: https://github.com/AutoGPTQ/AutoGPTQ/blob/caf343b1826301c15f90e2e119cabd0347acfcdf/auto_gptq/quantization/config.py#L60
@dataclass
class GPTQLinearConfig(ModelConfig):
    # quantization parameters
    bits: int = 4
    group_size: int = -1
    desc_act: bool = False

    # kernel selection
    # NOTE: default values select qlinear_cuda or qlinear_cuda_old kernel
    use_triton: bool = False
    disable_exllama: bool = True
    disable_exllamav2: bool = True
    use_qigen: bool = False
    use_marlin: bool = False
    use_tritonv2: bool = False

    # identifier
    linear_type: str = "gptq"


def custom_linear_repr(self):
    """Updated representation for AutoGPTQ QuantLinear class"""

    # desc_act is not an AutoGPTQ QuantLinear attribute,
    # we add in get_linear (from fms.modules.linear) after instantiating the object
    desc_act_str = f"desc_act={self.desc_act}, " if hasattr(self, "desc_act") else ""

    return (
        f"{self.__class__.__name__}"
        f"(in={self.infeatures}, out={self.outfeatures}, "
        f"bias={self.bias is not None}, "
        f"group={self.group_size}, {desc_act_str}"
        f"qtype={self.QUANT_TYPE})"
    )


def get_gptq_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config: Optional[Mapping[str, Any]] = None,
):
    gptq_config = GPTQLinearConfig(**linear_config)

    if not IS_AUTOGPTQ_AVAILABLE:
        raise ImportError("AutoGPTQ dynamic QuantLinear could not be imported")
    if gptq_config.desc_act:
        raise NotImplementedError(
            "Activation reordering (desc_act=True) not currently supported"
        )
    if gptq_config.use_marlin:
        raise NotImplementedError("Marlin kernels not currently supported")

    linear_class = dynamically_import_QuantLinear(
        use_triton=gptq_config.use_triton,
        desc_act=gptq_config.desc_act,
        group_size=gptq_config.group_size,
        bits=gptq_config.bits,
        disable_exllama=gptq_config.disable_exllama,
        disable_exllamav2=gptq_config.disable_exllamav2,
        use_qigen=gptq_config.use_qigen,
        use_marlin=gptq_config.use_marlin,
        use_tritonv2=gptq_config.use_tritonv2,
    )
    linear = linear_class(
        bits=gptq_config.bits,
        group_size=gptq_config.group_size,
        infeatures=in_features,
        outfeatures=out_features,
        bias=bias,
    )

    # provide AutoGPTQ QuantLinear attributes in nn.Linear form
    setattr(linear, "in_features", linear.infeatures)
    setattr(linear, "out_features", linear.outfeatures)
    setattr(linear, "desc_act", gptq_config.desc_act)

    # improve barebone AutoGPTQ representation (only one call needed)
    if linear.__class__.__repr__ != custom_linear_repr:
        linear.__class__.__repr__ = custom_linear_repr

    return linear


def shard_gptq_linear(
    tensor_values: Dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: Dict[str, LinearModuleShardingInfo],
) -> None:
    """
    Set up GPTQ quantization parameters to be sharded onto linear modules

                         |     GPU     |
    sharding  | qparam   | shard | dim |
    ----------+----------+-------+-----|
    colwise   | qweight  |   Y   |  1  |
              | bias     |   Y   |  0  |
              | scales   |   Y   |  1  |
              | qzeros   |   Y   |  1  |
              | g_idx    |   N   |  -  |
    ----------+----------+-------+-----|
    rowwise   | qweight  |   Y   |  0  |
              | bias     |   0   |  -  |
              | scales   |   Y   |  0  |
              | qzeros   |   Y   |  0  |
              | g_idx    |   Y   |  0  |
    """
    param_sharding_info: Dict[str, Dict[str, LinearParameterShardingInfo]] = {}
    for module_name, module_info in module_sharding_info.items():
        gptq_mod = module_info.linear_module
        params: Dict[str, LinearParameterShardingInfo] = {
            "qweight": LinearParameterShardingInfo(
                1 - module_info.sharding_dim, ShardType.SHARD
            ),
            "scales": LinearParameterShardingInfo(
                1 - module_info.sharding_dim, ShardType.SHARD
            ),
            "qzeros": LinearParameterShardingInfo(
                1 - module_info.sharding_dim, ShardType.SHARD
            ),
            "g_idx": LinearParameterShardingInfo(
                0,
                ShardType.CLONE if module_info.sharding_dim == 0 else ShardType.SHARD,
            ),
        }
        if gptq_mod.bias is not None:
            params["bias"] = LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.RANK0,
            )
        param_sharding_info[module_name] = params

    shard_base_linear(
        tensor_values, tp_module, module_sharding_info, param_sharding_info
    )

    # If desc_act=False, correct the g_idx
    for module_name, module_info in module_sharding_info.items():
        if module_info.linear_module.desc_act == False:
            g_idx_param = module_info.linear_module.g_idx
            module_info.linear_module.g_idx = g_idx_param - g_idx_param.min()


register_linear_type_to_module_map("gptq", get_gptq_linear)
register_linear_type_to_sharding_map("gptq", shard_gptq_linear)
