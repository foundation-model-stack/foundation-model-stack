import os
from typing import Union

from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAForCausalLM


def get_model(model_name_or_path: Union[str, os.PathLike]) -> HFAdaptedLLaMAForCausalLM:
    """
    Get a Huggingface adapted FMS model from an equivalent HF model

    Parameters
    ----------
    model_name_or_path: Union[str, os.PathLike]
        Either the name of the model in huggingface hub or the absolute path to
        the huggingface model

    Returns
    -------
    HFAdaptedLLaMAForCausalLM
        A Huggingface adapted FMS implementation of LLaMA
    """
    import torch
    from transformers import LlamaForCausalLM

    from fms.models.hf.utils import register_fms_models
    from fms.models.llama import convert_hf_llama

    register_fms_models()
    hf_model = LlamaForCausalLM.from_pretrained(model_name_or_path)
    fms_model = convert_hf_llama(hf_model).half()
    result_model: HFAdaptedLLaMAForCausalLM = HFAdaptedLLaMAForCausalLM.from_fms_model(
        fms_model,
        torch_dtype=torch.float16,
        # pad_token_id in fms is defaulted to -1
        # in generation, huggingface will add pad_tokens to the end of a sequence after the eos token is found for a
        # given sequence in the batch, if -1 is provided, our model won't be able to interpret it. We should be using
        # huggingface pad_token_id as that is where the model weights are coming from.
        pad_token_id=hf_model.config.pad_token_id,
    )
    return result_model
