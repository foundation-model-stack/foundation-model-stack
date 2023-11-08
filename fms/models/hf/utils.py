from typing import Union
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
import torch
import torch.nn as nn


def register_fms_models():
    """Register all FMS models with huggingface AutoModels. This will include Sandstone
    (AutoConfig, AutoModel, AutoModelForSeq2SeqLM) and Granite (AutoConfig, AutoModel, AutoModelForCausalLM)"""
    from fms.models.hf.llama.modeling_llama_hf import (
        HFAdaptedLLaMAConfig,
        HFAdaptedLLaMAHeadless,
        HFAdaptedLLaMAForCausalLM,
    )

    AutoConfig.register("hf_adapted_llama", HFAdaptedLLaMAConfig)
    AutoModel.register(HFAdaptedLLaMAConfig, HFAdaptedLLaMAHeadless)
    AutoModelForCausalLM.register(HFAdaptedLLaMAConfig, HFAdaptedLLaMAForCausalLM)

    from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import (
        HFAdaptedGPTBigCodeConfig,
        HFAdaptedGPTBigCodeHeadless,
        HFAdaptedGPTBigCodeForCausalLM,
    )

    AutoConfig.register("hf_adapted_gpt_bigcode", HFAdaptedGPTBigCodeConfig)
    AutoModel.register(HFAdaptedGPTBigCodeConfig, HFAdaptedGPTBigCodeHeadless)
    AutoModelForCausalLM.register(
        HFAdaptedGPTBigCodeConfig, HFAdaptedGPTBigCodeForCausalLM
    )


def mask_2d_to_3d(inp: torch.Tensor) -> torch.BoolTensor:
    """
    Produces a block-diagonal boolean attention mask matrix A where A[i,j]=True if tokens i and j are both pads,
    or both non-pads, False otherwise.
    ...
    Args
    ----
    inp : torch.Tensor
        Input batch of vocab indices. Expects shape [batch_size, sequence_length].
    ...
    Returns
    -------
    mask : torch.BoolTensor
        Mask tensor corresponding to inp. Will be of shape [batch_size, sequence_length, sequence_length].
    """
    is_pad = inp == 0
    mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
    return mask


def mask_2d_to_3d_bidirectional(
    decoder_input: Union[torch.BoolTensor, torch.LongTensor, torch.IntTensor],
    encoder_input: Union[torch.BoolTensor, torch.LongTensor, torch.IntTensor],
) -> torch.BoolTensor:
    """
    Produces a boolean attention mask matrix A where A[i,j]=True if tokens i and j are both pads, or both non-pads,
    from their respective inputs, False otherwise. If decoder_input is b*n1 and encoder_input is b*n2, output will be b*n1*n2.

    Note: There is a manual correction included in this function to avoid cross attending to nothing (in this case, we
    default to attend to everything)
    ...
    Args
    ----
    decoder_input : torch.Tensor
        Input batch of vocab indices. Expects shape [batch_size, sequence_length].
    encoder_input : torch.Tensor
        Input batch of vocab indices. Expects shape [batch_size, sequence_length].
    ...
    Returns
    -------
    mask : torch.BoolTensor
        Mask tensor corresponding to inp. Will be of shape [batch_size, sequence_length1, sequence_length2].
    """

    mask_encoder = encoder_input == 0
    mask_decoder = decoder_input == 0

    # The following was included to solve an issue where an all 0 sequence was provided which results
    # in a malformed cross attention mask that when provided to SDPA, will produce NaN. In the past,
    # an all false mask would just default to unmasked attention due to softmax shift invariance.
    _is_one_type_enc = mask_encoder.sum(1)
    _is_one_type_enc = _is_one_type_enc.eq(0) | _is_one_type_enc.eq(
        mask_encoder.size(1)
    )
    _is_one_type_dec = mask_decoder.sum(1)
    _is_one_type_dec = _is_one_type_dec.eq(0) | _is_one_type_dec.eq(
        mask_decoder.size(1)
    )

    # we need to correct if:
    #   (1) encoder is all one type and decoder has multiple types, then we need a correction
    #   (2) both encoder and decoder are one type, but those types don't match
    needs_correction_1 = _is_one_type_enc & ~_is_one_type_dec
    needs_correction_2 = (_is_one_type_enc & _is_one_type_dec) & mask_encoder[:, 0].ne(
        mask_decoder[:, 0]
    )
    needs_correction = needs_correction_1 | needs_correction_2
    mask_decoder = torch.where(
        needs_correction.unsqueeze(1), mask_encoder[:, 0].unsqueeze(1), mask_decoder
    )

    return mask_encoder.unsqueeze(1) == mask_decoder.unsqueeze(2)


def wrap(model: nn.Module, **override_config_kwargs) -> "HFModelArchitecture":
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
    from fms.models.hf.gpt_bigcode.modeling_gpt_bigcode_hf import (
        HFAdaptedGPTBigCodeForCausalLM,
    )
    from fms.models.hf.llama.modeling_llama_hf import HFAdaptedLLaMAForCausalLM

    __fms_to_hf_adapt_map = {
        "LLaMA": HFAdaptedLLaMAForCausalLM,
        "GPTBigCode": HFAdaptedGPTBigCodeForCausalLM,
    }

    model_name = model.__class__.__name__
    if model_name not in __fms_to_hf_adapt_map:
        raise ValueError(f"{model_name} is not one of {__fms_to_hf_adapt_map.keys()}")

    hf_adapted_cls = __fms_to_hf_adapt_map[model_name]
    return hf_adapted_cls.from_fms_model(model, **override_config_kwargs)
