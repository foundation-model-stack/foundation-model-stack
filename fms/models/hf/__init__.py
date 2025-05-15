# type: ignore
from torch import nn
from fms.models.hf.modeling_hf_adapter import HFModelArchitecture
from fms.models.gpt_bigcode import GPTBigCode, GPTBigCodeHeadless
from fms.models.hf.gpt_bigcode import HFAdaptedGPTBigCodeForCausalLM
from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import (
    HFAdaptedGPTBigCodeHeadless,
)
from fms.models.hf.granite.modeling_granite_hf import (
    HFAdaptedGraniteForCausalLM,
    HFAdaptedGraniteHeadless,
)
from fms.models.hf.llama import HFAdaptedLLaMAForCausalLM
from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAHeadless
from fms.models.hf.mixtral.modeling_mixtral_hf import (
    HFAdaptedMixtralForCausalLM,
    HFAdaptedMixtralHeadless,
)
from fms.models.hf.roberta.modeling_roberta_hf import (
    HFAdaptedRoBERTaForMaskedLM,
    HFAdaptedRoBERTaHeadless,
)
from fms.models.hf.utils import register_fms_models
from fms.models.llama import LLaMA
from fms.models.granite import Granite, GraniteHeadless
from fms.models.mixtral import Mixtral, MixtralHeadless
from fms.models.roberta import RoBERTa, RoBERTaHeadless


"""
mapping from an FMS model to its equivalent HF-Adapted model
"""
_fms_to_hf_adapt_map = {
    LLaMA: HFAdaptedLLaMAForCausalLM,
    Granite: HFAdaptedGraniteForCausalLM,
    GraniteHeadless: HFAdaptedGraniteHeadless,
    GPTBigCode: HFAdaptedGPTBigCodeForCausalLM,
    GPTBigCodeHeadless: HFAdaptedGPTBigCodeHeadless,
    RoBERTa: HFAdaptedRoBERTaForMaskedLM,
    RoBERTaHeadless: HFAdaptedRoBERTaHeadless,
    Mixtral: HFAdaptedMixtralForCausalLM,
    MixtralHeadless: HFAdaptedMixtralHeadless,
}

"""
list of all headless base HF-Adapted models used in registration
"""
_headless_models = [
    HFAdaptedGPTBigCodeHeadless,
    HFAdaptedGraniteHeadless,
    HFAdaptedLLaMAHeadless,
    HFAdaptedRoBERTaHeadless,
    HFAdaptedMixtralHeadless,
]

"""
list of all causal-lm HF-Adapted models used in registration
"""
_causal_lm_models = [
    HFAdaptedGPTBigCodeForCausalLM,
    HFAdaptedGraniteForCausalLM,
    HFAdaptedLLaMAForCausalLM,
    HFAdaptedMixtralForCausalLM,
]

"""
list of all masked-lm HF-Adapted models used in registration
"""
_masked_lm_models = [HFAdaptedRoBERTaForMaskedLM]


def to_hf_api(model: nn.Module, **override_config_kwargs) -> HFModelArchitecture:  # type: ignore
    """Wrap an FMS model, converting its API to one of and Huggingface model

    Parameters
    ----------
    model: nn.Module
        The FMS model to wrap (currently one of LLaMA or GPTBigCode)
    override_config_kwargs
        configuration parameters to override as a set of keyword arguments

    Returns
    -------
    HFModelArchitecture
        an HF adapted FMS model
    """
    from fms.models.hf import _fms_to_hf_adapt_map  # type: ignore

    register_fms_models()

    model_type = type(model)
    if model_type not in _fms_to_hf_adapt_map:
        raise ValueError(
            f"{model.__class__.__name__} is not one of {_fms_to_hf_adapt_map.keys()}"
        )

    hf_adapted_cls = _fms_to_hf_adapt_map[model_type]
    return hf_adapted_cls.from_fms_model(model, **override_config_kwargs)
