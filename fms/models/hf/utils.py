import os.path
from typing import Union

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

from fms.models import get_model as fms_get_model
from fms.models import list_variants


def register_fms_models():
    """Register all FMS models with huggingface AutoModels"""
    from fms.models.hf import _causal_lm_models, _headless_models, _masked_lm_models

    for model_cls in _headless_models:
        # register config
        AutoConfig.register(model_cls.config_class.model_type, model_cls.config_class)
        # register base headless model
        AutoModel.register(model_cls.config_class, model_cls)

    for model_cls in _causal_lm_models:
        # register causal lm model
        AutoModelForCausalLM.register(model_cls.config_class, model_cls)

    for model_cls in _masked_lm_models:
        # register masked lm models
        AutoModelForMaskedLM.register(model_cls.config_class, model_cls)


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


def to_hf_api(model: nn.Module, **override_config_kwargs) -> "HFModelArchitecture":
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
    from fms.models.hf import _fms_to_hf_adapt_map

    register_fms_models()

    model_type = type(model)
    if model_type not in _fms_to_hf_adapt_map:
        raise ValueError(
            f"{model.__class__.__name__} is not one of {_fms_to_hf_adapt_map.keys()}"
        )

    hf_adapted_cls = _fms_to_hf_adapt_map[model_type]
    return hf_adapted_cls.from_fms_model(model, **override_config_kwargs)


def get_model(
    model_id_or_path: Union[str, os.PathLike], device: Union[str, torch.device] = "cpu"
) -> nn.Module:
    """
    get an FMS model from a huggingface checkpoint

    Parameters
    ----------
    model_id_or_path: Union[str, os.PathLike]
        The huggingface hub model id or a local path. If the local path exists, the model will be loaded directly from
        the local path, otherwise the huggingface cache will be checked. If the huggingface cache does not contain the
        model, then the weights will be downloaded and stored into the huggingface cache
    device: Union[str, torch.device]
        the device to load the model weights to

    Returns
    -------
    nn.Module
        an fms equivalent implementation of an HF model
    """
    if not os.path.exists(model_id_or_path):
        model_id_or_path = snapshot_download(repo_id=model_id_or_path)

    config = AutoConfig.from_pretrained(model_id_or_path)

    architecture = config.architectures[0]
    params = {
        "model_path": model_id_or_path,
        "source": "hf",
        "device_type": device.type if isinstance(device, torch.device) else device,
    }

    if architecture == "LlamaForCausalLM":
        params["architecture"] = "llama"
        params["attn_bias"] = getattr(config, "attention_bias", False)
        params["mlp_bias"] = getattr(config, "mlp_bias", False)
        params["kv_heads"] = config.num_key_value_heads
        params["norm_eps"] = config.rms_norm_eps
        params["multiple_of"] = 1
        inner_dim = config.intermediate_size
        max_expected_seq_len = config.max_position_embeddings
    elif architecture == "GPTBigCodeForCausalLM":
        params["architecture"] = "gpt_bigcode"
        params["ln_eps"] = config.layer_norm_epsilon
        params["multiquery_attn"] = config.multi_query
        inner_dim = config.n_inner
        max_expected_seq_len = config.n_positions
    else:
        raise ValueError(
            "FMS model implementations currently only support LlamaForCausalLM and GPTBigCodeForCausalLM"
        )

    # infer common params
    params["variant"] = list_variants(params["architecture"])[0]
    params["src_vocab_size"] = config.vocab_size
    params["emb_dim"] = config.hidden_size
    params["nheads"] = config.num_attention_heads
    params["nlayers"] = config.num_hidden_layers
    params["hidden_grow_factor"] = inner_dim / config.hidden_size
    params["max_expected_seq_len"] = max_expected_seq_len
    params["tie_heads"] = config.tie_word_embeddings

    return fms_get_model(**params)
