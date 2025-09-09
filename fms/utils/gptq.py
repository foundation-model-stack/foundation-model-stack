import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

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
    from auto_gptq.utils.import_utils import (  # type: ignore[import-untyped,import-not-found]
        dynamically_import_QuantLinear,
    )

    IS_AUTOGPTQ_AVAILABLE = True
except ImportError:
    IS_AUTOGPTQ_AVAILABLE = False


def check_if_gptq(extra_args: Mapping) -> bool:
    linear_config = extra_args.get("linear_config", None)
    if linear_config:
        linear_type = linear_config.get("linear_type", None)
        if linear_type and isinstance(linear_type, str):
            return "gptq" in linear_type
    return False


# simplified from AutoGPTQ quantization config
# see: https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/quantization/config.py#L60
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

    # linear module identifiers
    linear_type: str = "gptq"
    module_name: str = ""


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
    linear_config: Mapping[str, Any],
):
    """Construct and return autogptq linear module"""

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
    tensor_values: dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: dict[str, LinearModuleShardingInfo],
) -> Optional[set]:
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
    param_sharding_info: dict[str, dict[str, LinearParameterShardingInfo]] = {}
    for module_name, module_info in module_sharding_info.items():
        gptq_mod = module_info.linear_module
        params: dict[str, LinearParameterShardingInfo] = {
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
        if hasattr(gptq_mod, "bias") and gptq_mod.bias is not None:
            params["bias"] = LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.RANK0,
            )
        param_sharding_info[module_name] = params

    unused_keys = shard_base_linear(
        tensor_values, tp_module, module_sharding_info, param_sharding_info
    )

    # If desc_act=False, correct the g_idx
    for module_name, module_info in module_sharding_info.items():
        if not module_info.linear_module.desc_act:
            g_idx_param = getattr(module_info.linear_module, "g_idx")
            setattr(module_info.linear_module, "g_idx", g_idx_param - g_idx_param.min())

    return unused_keys


register_linear_type_to_module_map("gptq", get_gptq_linear)
register_linear_type_to_sharding_map("gptq", shard_gptq_linear)


class GPTQLinearCPU(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        config: GPTQLinearConfig,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = config.bits
        self.group_size = config.group_size if config.group_size != -1 else in_features
        self.desc_act = config.desc_act

        if self.bits not in [4]:
            raise NotImplementedError(
                "GPTQLinear for CPU only supports 4 bits quantization."
            )
        if in_features % self.group_size != 0:
            raise ValueError("`in_features` must be divisible by `group_size`.")
        if in_features % 32 or out_features % 32:
            raise ValueError("`in_features` and `out_features` must be divisible by 32")
        if self.desc_act:
            raise NotImplementedError(
                "GPTQLinear for CPU does not support activation reordering (`desc_act`)"
            )

        # Register quantization parameters
        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features // 32 * self.bits, out_features),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(in_features / self.group_size),
                    out_features // 32 * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(in_features / self.group_size), out_features),
                dtype=torch.float16,
            ),
        )
        self.register_buffer("g_idx", torch.zeros(in_features, dtype=torch.int32))
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=torch.float16)
            )
        else:
            self.bias = None

    def unpack_qparam(
        self,
        qparam: torch.Tensor,
        bits: int = 4,
        transpose: bool = False,
    ) -> torch.Tensor:
        """Unpack GPTQ tensor of quantized qweight or qzeros, expanding
        input matrix "32 // bits" times along originally packed dimension.

        - qweights are packed by GPTQ along dim=0 and expanded row-wise
        - qzeros are packed by GPTQ along dim=1 and expanded column-wise
        (matrix is transposed before and after packing)

        qweight packed size: [in_feat * bits // 32, out_feat]
        qzeros packed size: [n_grp, out_feat * bits // 32]
                            with n_grp = in_feat / grp_size

        qweight unpacked size: [in_feat, out_feat]
        qzeros unpacked size: [n_grp, out_feat]

        Parameters
        ----------
        qparam : torch.Tensor
            a packed quantization parameter (qweight or qzeros)
        bits : int
            number of quantization bits
        transpose : bool
            apply transpose before and after unpacking (needed by qzeros)

        Return
        ------
        qparam_unpacked : torch.Tensor
            the quantization parameter (qweight or qzeros) unpacked
        """

        if transpose:
            qparam = qparam.t()

        device = qparam.device
        unpack_size = 32 // bits
        qparam_unpacked = torch.zeros(
            (qparam.size(0) * unpack_size, qparam.size(1)),
            dtype=torch.int,
            device=device,
        )

        for j in range(unpack_size):
            qparam_unpacked[j::unpack_size, :] = (qparam >> (j * bits)) & 0xF
        return qparam_unpacked.t() if transpose else qparam_unpacked

    def dequantize(self, qweights_unpacked, scales):
        """Dequantize unpacked weight tensor

        qweights_unpacked: [in_feat, out_feat]
        scales: [n_groups, out_feat]
        qzeros: assume symmetric quantization => constant (8)
        g_idx: activation reordering not supported
        """

        zp = 8
        in_feat, _ = qweights_unpacked.size()
        n_grp, _ = scales.size()
        return torch.mul(
            (qweights_unpacked - zp).to(torch.float16),
            scales.repeat_interleave(in_feat // n_grp, dim=0),
        )

    def forward(self, x):
        qweights_unpacked = self.unpack_qparam(self.qweight, self.bits)
        weights = self.dequantize(qweights_unpacked, self.scales)
        x = torch.matmul(x.to(weights.dtype), weights)
        if self.bias is not None:
            x.add_(self.bias)
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, group={self.group_size})"
        )


def get_gptq_cpu_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config: Optional[Mapping[str, Any]] = None,
):
    if linear_config is not None:
        gptq_config = GPTQLinearConfig(**linear_config)
    else:
        raise ValueError("GPTQLinearConfig requires a linear config")
    if gptq_config.desc_act:
        raise NotImplementedError("Activation reordering (desc_act=True) not supported")
    linear = GPTQLinearCPU(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        config=gptq_config,
    )
    return linear


register_linear_type_to_module_map("gptq_cpu", get_gptq_cpu_linear)
register_linear_type_to_sharding_map("gptq_cpu", shard_gptq_linear)
