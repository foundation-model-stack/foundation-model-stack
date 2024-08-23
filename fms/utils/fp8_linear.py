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

from torchao.float8.float8_utils import to_fp8_saturated
HAS_FBGEMM_EXPERIMENTAL = False
try:
    import fbgemm_gpu.experimental.gen_ai
    HAS_FBGEMM_EXPERIMENTAL = True
except:
    pass

@dataclass
class Fp8LinearConfig(ModelConfig):
    activation_quantization: str = "dynamic-per-tensor"
    activation_quantization_ub: float = None
    weight_quantization: str = "static-per-tensor"
    linear_type: str = "auto_fp8" # or "fbgemm_fp8"
    use_fast_accum: bool = True

class Fp8InferenceLinearBase(torch.nn.Module):
    def __init__(
        self,
        quant_config,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.quant_type = quant_config.linear_type
        self.activation_casting = quant_config.activation_quantization
        self.weight_casting = quant_config.weight_quantization
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    # Calculate the scale as dtype max divided by absmax
    scale = finfo.max / x.abs().max().clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    return x_scl_sat.to(dtype), scale.float().reciprocal()

class AutoFp8InferenceLinear(Fp8InferenceLinearBase):
    def __init__(
        self,
        quant_config,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        # Construct the superclass this will create dummy weights and biases
        super().__init__(quant_config, in_features, out_features)
        # self.use_fast_accum = quant_config.use_fast_accum

        assert (self.activation_casting == "static-per-tensor" 
                or self.activation_casting == "dynamic-per-tensor")
        if self.activation_casting == "static-per-tensor":
            self.register_parameter(
                "input_scale",
                nn.Parameter(torch.tensor(0.0, dtype=torch.float32)),
            )
        else:
            self.input_scale = None

        # N x K
        self.register_parameter(
            "weight",
            nn.Parameter(torch.empty((self.out_features, self.in_features), dtype=torch.float8_e4m3fn)),
        )
        
        # 1
        assert self.weight_casting == "static-per-tensor"
        self.register_parameter(
            "weight_scale",
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32)), # per-tensor
        )
        
        if bias:
            self.register_parameter("bias", nn.Parameter(torch.zeros((self.out_features,), dtype=torch.bfloat16)))
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        wq = self.weight
        wscale = self.weight_scale
        
        bs = None
        # FIXME: is this strictly necessary?
        if input.dim() > 2:
            bs = input.shape[0]
            input = input.reshape(-1, input.shape[2])

        if self.activation_casting == "dynamic-per-tensor":
            aq, ascale = to_float8(input)
        elif self.activation_casting == "static-per-tensor":
            ascale = self.input_scale
            aq = to_fp8_saturated((input * ascale.reciprocal()), torch.float8_e4m3fn) # scale is inverse for torchao.float8
        else:
            raise NotImplementedError("only per-tensor activation quantization supported in AutoFP8")

        output = torch._scaled_mm(aq,
                                  wq.T,
                                  out_dtype=input.dtype,
                                  scale_a=ascale, # per-tensor
                                  scale_b=wscale, # per-tensor
                                  bias=self.bias)
        
        if bs is not None:
            output = output.reshape(bs, -1, output.shape[1])

        return output


class Fp8InferenceLinear(Fp8InferenceLinearBase):
    """
    This mocks nn.Linear and supports FP8 inference
    using Fbgemm gen_ai kernels
    Supported forms of inference:
        - FP8 inference dynamic activation per-token quantization with upperbound 
          and static per-channel weight quantization
        - FP8 inference static activation per-token quantization and static 
          per-channel weight quantization
    """

    def __init__(
        self,
        quant_config,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        if not HAS_FBGEMM_EXPERIMENTAL:
            raise ImportError("Failed to import fbgemm_gpu.experimental.gen_ai. Please install fbgemm_gpu!")
        # Construct the superclass this will create dummy weights and biases
        super().__init__(quant_config, in_features, out_features)
        self.use_fast_accum = quant_config.use_fast_accum
  
        if self.activation_casting == "static-per-tensor":
            self.register_parameter(
                "input_scale",
                nn.Parameter(torch.tensor(0.0, dtype=torch.float32)),
            )
        else:
            self.input_scale = None

        # Fbgemm activation scale upperbound
        if not (quant_config.activation_quantization_ub is None):
            self.register_parameter(
                "activation_quantization_ub",
                nn.Parameter(torch.tensor(quant_config.activation_quantization_ub)),
            )
        else:
            self.activation_quantization_ub = None

        # N x K
        self.register_parameter(
            "weight",
            nn.Parameter(torch.empty((self.out_features, self.in_features), dtype=torch.float8_e4m3fn)),
        )
        
        # 1 or N
        self.register_parameter(
            "weight_scale",
            nn.Parameter(
                (torch.tensor(0.0, dtype=torch.float32) 
                    if self.weight_casting == "static-per-tensor"
                    else torch.empty((self.out_features,), dtype=torch.float32)
                )
            ),
        )
        
        if bias:
            self.register_parameter(
                "bias",
                nn.Parameter(torch.zeros((self.out_features,), dtype=torch.bfloat16)),
            )
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        wq = self.weight
        wscale = self.weight_scale
        
        bs = None
        # FIXME: is this strictly necessary?
        if input.dim() > 2:
            bs = input.shape[0]
            input = input.reshape(-1, input.shape[2])

        if self.activation_casting == "dynamic-per-row":
            aq, ascale = torch.ops.fbgemm.quantize_fp8_per_row(input, scale_ub=self.activation_quantization_ub)
        elif self.activation_casting == "dynamic-per-tensor":
            aq, ascale = torch.ops.fbgemm.quantize_fp8_per_tensor(input, scale_ub=self.activation_quantization_ub)
        elif self.activation_casting == "static-per-tensor":
            ascale = self.input_scale
            aq = to_fp8_saturated((input * ascale.reciprocal()), torch.float8_e4m3fn) # scale is inverse for torchao.float8
        else:
            raise NotImplementedError

        # per-tensor A and per-tensor B
        if (self.activation_casting == "static-per-tensor" or 
            self.activation_casting == "dynamic-per-tensor"): 
            assert "static-per-tensor" == self.weight_casting
            output = torch.ops.fbgemm.f8f8bf16_tensorwise(
                aq, wq, 
                ascale * wscale, # per-tensor 
            )
            if self.bias is not None:
                output += self.bias
        else: # per-row A and per-row B
            assert "static-per-row" == self.weight_casting
            assert "dynamic-per-row" == self.activation_casting
            
            output = torch.ops.fbgemm.f8f8bf16_rowwise(
                        aq, wq, 
                        ascale, # per-row
                        wscale, # per-row
                        self.bias,
                        use_fast_accum=self.use_fast_accum,
                    )
            
            # # DEBUG implementation with torch scaled mm
            # output = torch._scaled_mm(aq, wq.T,
            #                           out_dtype=input.dtype,
            #                           scale_a=torch.tensor(1.0, dtype=torch.float32, device=aq.device), 
            #                           scale_b=torch.tensor(1.0, dtype=torch.float32, device=aq.device))
            
            # scales = ascale.view((output.shape[0],1)) @ wscale.view(1, output.shape[1])
            # # print(f"{scales=}, {scales.shape=}")
            # output = (output * scales).to(torch.bfloat16)
            # print(f"{output=}, {output.shape=}")
        if bs is not None:
            output = output.reshape(bs, -1, output.shape[1])

        return output

def get_fp8_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config: Optional[Mapping[str, Any]] = None
):
    fp8_linear_config = Fp8LinearConfig(**linear_config)
    map_type_to_linear = {
        "auto_fp8": AutoFp8InferenceLinear,
        "fbgemm_fp8": Fp8InferenceLinear,
    }
    
    linear_class = map_type_to_linear[fp8_linear_config.linear_type]

    linear = linear_class(
        quant_config=fp8_linear_config,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
    )
    assert hasattr(linear, "in_features") and hasattr(linear, "out_features")
    return linear


def shard_auto_fp8_linear(
    tensor_values: Dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: Dict[str, LinearModuleShardingInfo],
) -> None:
    param_sharding_info: Dict[str, Dict[str, LinearParameterShardingInfo]] = {}
    for module_name, module_info in module_sharding_info.items():
        linear_module = module_info.linear_module
        params: Dict[str, LinearParameterShardingInfo] = {
            "weight_scale": LinearParameterShardingInfo( # per-tensor
                0,
                ShardType.CLONE,
            ),
            "weight": LinearParameterShardingInfo(
                module_info.sharding_dim, ShardType.SHARD
            ),
        }
        if linear_module.bias is not None:
            params["bias"] = LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.RANK0,
            )
        if linear_module.input_scale is not None:
            params["input_scale"] = LinearParameterShardingInfo( # per-tensor
                0,
                ShardType.CLONE,
            )
        param_sharding_info[module_name] = params

    shard_base_linear(
        tensor_values, tp_module, module_sharding_info, param_sharding_info
    )

def shard_fbgemm_fp8_linear(
    tensor_values: Dict[str, torch.Tensor],
    tp_module: TPModule,
    module_sharding_info: Dict[str, LinearModuleShardingInfo],
) -> None:
    param_sharding_info: Dict[str, Dict[str, LinearParameterShardingInfo]] = {}
    for module_name, module_info in module_sharding_info.items():
        linear_module = module_info.linear_module
        params: Dict[str, LinearParameterShardingInfo] = {
            # FIXME: should either shard or clone depending on the checkpoint weight_scale
            "weight_scale": LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.CLONE,
            ),
            "weight": LinearParameterShardingInfo(
                module_info.sharding_dim, ShardType.SHARD
            ),
        }
        if linear_module.bias is not None:
            params["bias"] = LinearParameterShardingInfo(
                module_info.sharding_dim,
                ShardType.SHARD if module_info.sharding_dim == 0 else ShardType.RANK0,
            )
        if linear_module.input_scale is not None:
            params["input_scale"] = LinearParameterShardingInfo( # per-tensor
                0,
                ShardType.CLONE,
            )
        param_sharding_info[module_name] = params

    shard_base_linear(
        tensor_values, tp_module, module_sharding_info, param_sharding_info
    )

register_linear_type_to_module_map("auto_fp8", get_fp8_linear)
register_linear_type_to_sharding_map("auto_fp8", shard_auto_fp8_linear)

register_linear_type_to_module_map("fbgemm_fp8", get_fp8_linear)
register_linear_type_to_sharding_map("fbgemm_fp8", shard_fbgemm_fp8_linear)
