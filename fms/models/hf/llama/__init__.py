import os
from typing import Union
from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAForCausalLM


def get_model(model_name_or_path: Union[str, os.PathLike]) -> HFAdaptedLLaMAForCausalLM:
    """
    Get a Huggingface adapted FMS model from an equivalent HF model

    Parameters
    ----------
    model_path_or_name: Union[str, os.PathLike]
        Either the name of the model in huggingface hub or the absolute path to
        the huggingface model

    Returns
    -------
    HFAdaptedLLaMAForCausalLM
        A Huggingface adapted FMS implementation of LLaMA
    """
    import torch
    from fms.models.hf.utils import register_fms_models
    from fms.models.llama import convert_hf_llama
    from transformers import LlamaForCausalLM

    register_fms_models()
    hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path)
    fms_model = convert_hf_llama(hf_model).half()
    result_model: HFAdaptedLLaMAForCausalLM = HFAdaptedLLaMAForCausalLM.from_fms_model(
        fms_model, torch_dtype=torch.float16
    )
    return result_model
