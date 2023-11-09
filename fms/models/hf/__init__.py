from fms.models.hf.utils import to_hf_api, register_fms_models

from fms.models.hf.gpt_bigcode import HFAdaptedGPTBigCodeForCausalLM
from fms.models.hf.llama import HFAdaptedLLaMAForCausalLM
from fms.models.gpt_bigcode import GPTBigCode, GPTBigCodeHeadless
from fms.models.llama import LLaMA
from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import (
    HFAdaptedGPTBigCodeHeadless,
)
from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAHeadless

"""
mapping from an FMS model to its equivalent HF-Adapted model
"""
_fms_to_hf_adapt_map = {
    LLaMA: HFAdaptedLLaMAForCausalLM,
    GPTBigCode: HFAdaptedGPTBigCodeForCausalLM,
    GPTBigCodeHeadless: HFAdaptedGPTBigCodeHeadless,
}

"""
list of all headless base HF-Adapted models used in registration 
"""
_headless_models = [HFAdaptedGPTBigCodeHeadless, HFAdaptedLLaMAHeadless]

"""
list of all causal-lm HF-Adapted models used in registration 
"""
_causal_lm_models = [HFAdaptedGPTBigCodeForCausalLM, HFAdaptedLLaMAForCausalLM]
