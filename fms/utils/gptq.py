from dataclasses import dataclass
from fms.utils.config import ModelConfig


def custom_linear_repr(self):
    """Updated representation for AutoGPTQ QuantLinear class"""

    # desc_act is not a GPTQ QuantLinear attribute,
    # it is added in get_linear after instantiating the object
    desc_act_str = f"desc_act={self.desc_act}, " if hasattr(self, "desc_act") else ""

    return (
        f"{self.__class__.__name__}"
        f"(in={self.infeatures}, out={self.outfeatures}, "
        f"bias={self.bias is not None}, "
        f"group={self.group_size}, {desc_act_str}"
        f"qtype={self.QUANT_TYPE})"
    )


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
