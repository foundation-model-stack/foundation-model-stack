from fms.utils.gptq import GPTQConfig, custom_linear_repr
from typing import Optional, Mapping, Any

try:
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
    IS_AUTOGPTQ_AVAILABLE=True
except:
    IS_AUTOGPTQ_AVAILABLE=False


def get_gptq_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    linear_config: Optional[Mapping[str, Any]] = None,
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

    # AutoGPTQ QuantLinear
    setattr(linear, "in_features", linear.infeatures)
    setattr(linear, "out_features", linear.outfeatures)
    setattr(linear, "desc_act", gptq_config.desc_act)

    # improve barebone AutoGPTQ representation (only one call needed)
    if linear.__class__.__repr__ != custom_linear_repr:
        linear.__class__.__repr__ = custom_linear_repr

    return linear
