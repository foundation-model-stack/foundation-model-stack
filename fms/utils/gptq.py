from dataclasses import dataclass
from typing import Mapping, Any
from fms.utils.config import ModelConfig

try:
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
    IS_AUTOGPTQ_AVAILABLE=True
except:
    IS_AUTOGPTQ_AVAILABLE=False


# simplified from AutoGPTQ quantization config
# see: https://github.com/AutoGPTQ/AutoGPTQ/blob/caf343b1826301c15f90e2e119cabd0347acfcdf/auto_gptq/quantization/config.py#L60
@dataclass
class GPTQConfig(ModelConfig):
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
    gptq_config = GPTQConfig(**linear_config)

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
    setattr(linear, "in_features", lambda: linear.infeatures)
    setattr(linear, "out_features", lambda: linear.outfeatures)
    setattr(linear, "desc_act", gptq_config.desc_act)

    # improve barebone AutoGPTQ representation (only one call needed)
    if linear.__class__.__repr__ != custom_linear_repr:
        linear.__class__.__repr__ = custom_linear_repr

    return linear
