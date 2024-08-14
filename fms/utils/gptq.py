from dataclasses import dataclass
from typing import Mapping, Any
import torch
import torch.nn as nn
from fms.utils.config import ModelConfig
from fms.modules.tp import TPModule
from fms.modules.linear import (
    register_linear_type_to_module_map,
    register_linear_type_to_sharding_map,
    shard_base_linear,
)

try:
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
    IS_AUTOGPTQ_AVAILABLE=True
except:
    IS_AUTOGPTQ_AVAILABLE=False


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
    linear_config: Mapping[str, Any] | None = None,
):
    gptq_config = GPTQLinearConfig(**linear_config)

    if not IS_AUTOGPTQ_AVAILABLE:
        raise ImportError("AutoGPTQ dynamic QuantLinear could not be imported")
    if gptq_config.desc_act:
        raise NotImplementedError("Activation reordering (desc_act=True) not currently supported")
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
    tensor_values: dict[str, torch.Tensor],
    tp_module: TPModule,
    modules: list[str],
    name_to_module: dict[str, nn.Module],
    module_base_shard_dim: dict[str, int],
    max_partition: dict [str, str],
) -> None:
    """
    Set up GPTQ quantization parameters to be sharded onto linear modules

                         |     GPU     |
    module    | qparam   | shard | dim |
    ----------+----------+-------+-----|
    QKV, w1   | qweight  |   Y   |  1  |
              | bias     |   Y   |  0  |
              | scales   |   Y   |  1  |
              | qzeros   |   Y   |  1  |
              | g_idx    |   N   |  -  |
    ----------+----------+-------+-----|
    dense, w2 | qweight  |   Y   |  0  |
              | bias     |   N   |  -  |
              | scales   |   Y   |  0  |
              | qzeros   |   Y   |  0  |
              | g_idx    |   Y   |  0  |
    """

    params = ["qweight", "scales", "qzeros", "g_idx"]
    if tp_module.use_bias:
        params.append("bias")

    # GPTQ qweights are transposed compared to nn.Linear weights
    module_base_shard_dim_gptq = {}
    for name, dim in module_base_shard_dim.items():
        module_base_shard_dim_gptq[name] = 1 - dim

    # TODO: improve this
    # List of tuples (module, param) that won't be sharded
    if "qkv_fused" in modules:  # MHA fused
        unsharded = [
            ("qkv_fused", "g_idx"),
            ("dense", "bias"),
        ]
    if "query" in modules:  # MHA unfused
        unsharded = [
            ("query", "g_idx"),
            ("key", "g_idx"),
            ("value", "g_idx"),
            ("dense", "bias")
        ]
    if "w1" in modules:  # FFN (no fusion)
        unsharded = [
            ("w1", "g_idx"),
            ("w2", "bias"),
        ]

    shard_base_linear(
        tensor_values,
        tp_module,
        modules,
        params,
        name_to_module,
        module_base_shard_dim_gptq,
        max_partition,
        unsharded,
    )


register_linear_type_to_module_map("gptq", get_gptq_linear)
register_linear_type_to_sharding_map("gptq", shard_gptq_linear)