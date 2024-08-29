# type: ignore
from fms.models.gpt_bigcode import GPTBigCode, GPTBigCodeHeadless
from fms.models.hf.gpt_bigcode import HFAdaptedGPTBigCodeForCausalLM
from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import (
    HFAdaptedGPTBigCodeHeadless,
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
from fms.models.hf.utils import as_fms_model, register_fms_models, to_hf_api
from fms.models.llama import LLaMA
from fms.models.mixtral import Mixtral, MixtralHeadless
from fms.models.roberta import RoBERTa, RoBERTaHeadless


"""
mapping from an FMS model to its equivalent HF-Adapted model
"""
_fms_to_hf_adapt_map = {
    LLaMA: HFAdaptedLLaMAForCausalLM,
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
    HFAdaptedLLaMAHeadless,
    HFAdaptedRoBERTaHeadless,
    HFAdaptedMixtralHeadless,
]

"""
list of all causal-lm HF-Adapted models used in registration 
"""
_causal_lm_models = [
    HFAdaptedGPTBigCodeForCausalLM,
    HFAdaptedLLaMAForCausalLM,
    HFAdaptedMixtralForCausalLM,
]

"""
list of all masked-lm HF-Adapted models used in registration
"""
_masked_lm_models = [HFAdaptedRoBERTaForMaskedLM]
