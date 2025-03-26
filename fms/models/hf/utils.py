import os.path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from torch._C._distributed_c10d import ProcessGroup
from transformers import (  # type: ignore
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

from fms.models import get_model, list_variants


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


def mask_2d_to_3d(inp: torch.Tensor) -> torch.Tensor:
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
    mask : torch.Tensor
        Mask tensor corresponding to inp. Will be of shape [batch_size, sequence_length, sequence_length].
    """
    is_pad = inp == 0
    mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
    return mask


def mask_2d_to_3d_bidirectional(
    decoder_input: Union[torch.BoolTensor, torch.LongTensor, torch.IntTensor],
    encoder_input: Union[torch.BoolTensor, torch.LongTensor, torch.IntTensor],
) -> torch.Tensor:
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
    mask : torch.Tensor
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


def _infer_model_configuration(
    model_id_or_path: str | os.PathLike,
    download_weights: bool = True,
) -> Dict[str, Any]:
    # if the path does not exist, download it from huggingface and get the local path
    if not os.path.exists(model_id_or_path):
        from huggingface_hub import snapshot_download  # type: ignore

        # in the case we don't want to download the weights, but just create the model from scratch, we will only allow config.json
        if download_weights:
            allow_patterns = ["*config.json", "tokenizer*", "special_tokens_map.json"]

            # mixtral saves safetensors expert sharded, so we will need their pt checkpoints
            # ideally this should be fixed in the adapter in the future
            ignore_patterns = None
            if isinstance(model_id_or_path, str) and model_id_or_path.startswith(
                "mistralai/Mixtral"
            ):
                ignore_patterns = ["*.safetensors"]
                allow_patterns.append("*.pt")
            else:
                allow_patterns.append("*.safetensors*")
        else:
            allow_patterns = ["config.json"]
            ignore_patterns = None

        model_path = snapshot_download(
            repo_id=str(model_id_or_path),
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns,
            cache_dir=os.environ.get("HF_HOME", None),
        )
    else:
        model_path = str(model_id_or_path)

    config = AutoConfig.from_pretrained(model_path)

    architecture = config.architectures[0]
    config_params = {}

    if architecture == "LlamaForCausalLM":
        inner_dim = config.intermediate_size
        architecture = "llama"
        config_params["attn_bias"] = getattr(config, "attention_bias", False)
        config_params["mlp_bias"] = getattr(config, "mlp_bias", False)
        config_params["kvheads"] = config.num_key_value_heads
        config_params["norm_eps"] = config.rms_norm_eps
        config_params["multiple_of"] = 1
        config_params["emb_dim"] = config.hidden_size
        config_params["max_expected_seq_len"] = config.max_position_embeddings
    elif architecture == "GPTBigCodeForCausalLM":
        inner_dim = config.n_inner
        architecture = "gpt_bigcode"
        config_params["ln_eps"] = config.layer_norm_epsilon
        config_params["multiquery_attn"] = config.multi_query
        config_params["emb_dim"] = config.hidden_size
        config_params["max_expected_seq_len"] = config.n_positions
    elif architecture == "MixtralForCausalLM":
        inner_dim = config.intermediate_size
        architecture = "mixtral"
        config_params["dim"] = config.hidden_size
        config_params["hidden_dim"] = inner_dim
        config_params["norm_eps"] = config.rms_norm_eps
        config_params["kv_heads"] = config.num_key_value_heads
        config_params["num_experts"] = config.num_local_experts
        config_params["top_k_experts"] = config.num_experts_per_tok
        config_params["rope_base"] = config.rope_theta
        config_params["max_expected_seq_len"] = config.max_position_embeddings
    elif architecture == "RobertaForMaskedLM":
        inner_dim = config.intermediate_size
        architecture = "roberta"
        config_params["emb_dim"] = config.hidden_size
        config_params["pad_id"] = config.pad_token_id
        config_params["max_pos"] = config.max_position_embeddings - 2
        config_params["p_dropout"] = config.hidden_dropout_prob
        config_params["norm_eps"] = config.layer_norm_eps
        config_params["activation_fn"] = config.hidden_act
    elif architecture == "RobertaForQuestionAnswering":
        inner_dim = config.intermediate_size
        architecture = "roberta_question_answering"
        config_params["emb_dim"] = config.hidden_size
        config_params["pad_id"] = config.pad_token_id
        config_params["max_pos"] = config.max_position_embeddings - 2
        config_params["p_dropout"] = config.hidden_dropout_prob
        config_params["norm_eps"] = config.layer_norm_eps
        config_params["activation_fn"] = config.hidden_act
    elif architecture == "GraniteForCausalLM":
        inner_dim = config.intermediate_size
        architecture = "granite"
        config_params["attn_bias"] = getattr(config, "attention_bias", False)
        config_params["mlp_bias"] = getattr(config, "mlp_bias", False)
        config_params["kvheads"] = config.num_key_value_heads
        config_params["norm_eps"] = config.rms_norm_eps
        config_params["multiple_of"] = 1
        config_params["emb_dim"] = config.hidden_size
        config_params["max_expected_seq_len"] = config.max_position_embeddings
        config_params["residual_multiplier"] = config.residual_multiplier
        config_params["attention_multiplier"] = config.attention_multiplier
        config_params["logits_scaling"] = config.logits_scaling
        config_params["embedding_multiplier"] = config.embedding_multiplier
    elif architecture == "BambaForCausalLM":
        inner_dim = config.intermediate_size
        architecture = "bamba"
        config_params["kvheads"] = config.num_key_value_heads
        config_params["p_dropout"] = config.attention_dropout
        config_params["activation_fn"] = config.hidden_act
        config_params["emb_dim"] = config.hidden_size
        config_params["chunk_size"] = config.mamba_chunk_size
        config_params["use_conv_bias"] = config.mamba_conv_bias
        config_params["conv_kernel"] = config.mamba_d_conv
        config_params["head_dim"] = config.mamba_d_head
        config_params["state_size"] = config.mamba_d_state
        config_params["mamba_expand"] = config.mamba_expand
        config_params["n_groups"] = config.mamba_n_groups
        config_params["mamba_n_heads"] = config.mamba_n_heads
        config_params["use_bias"] = config.mamba_proj_bias
        config_params["norm_eps"] = config.rms_norm_eps
    else:
        raise ValueError(
            "FMS model implementations currently only support LlamaForCausalLM, GPTBigCodeForCausalLM, MixtralForCausalLM, RobertaForMaskedLM and GraniteForCausalLM"
        )

    # infer common params
    config_params["src_vocab_size"] = config.vocab_size
    config_params["nheads"] = config.num_attention_heads
    config_params["nlayers"] = config.num_hidden_layers
    config_params["hidden_grow_factor"] = inner_dim / config.hidden_size
    config_params["tie_heads"] = config.tie_word_embeddings

    # infer get_model params
    config_params["architecture"] = architecture
    config_params["variant"] = list_variants(architecture)[0]
    config_params["model_path"] = model_path if download_weights else None
    return config_params


def as_fms_model(
    model_id_or_path: Union[str, os.PathLike],
    device_type: str = "cpu",
    data_type: Optional[Union[str, torch.dtype]] = None,
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    group: Optional[ProcessGroup] = None,
    initialize_model_with_weights: bool = True,
) -> nn.Module:
    """
    get an FMS model from a huggingface checkpoint

    Parameters
    ----------
    model_id_or_path: Union[str, os.PathLike]
        The huggingface hub model id or a local path. If the local path exists, the model will be loaded directly from
        the local path, otherwise the huggingface cache will be checked. If the huggingface cache does not contain the
        model, then the weights will be downloaded and stored into the huggingface cache
    device_type: where to load the model
    distributed_strategy: None, 'fsdp', 'hsdp', 'tp', or 'mp'.
    checkpoint_sharding: how the checkpoint files are sharded: None, 'tp',
                'fsdp', or 'layer'. If None, guess based on files.
    group: ProcessGroup The PG to use for any model distribution
    initialize_model_with_weights: bool
        If True, will download the weights for the model and load them into the fms model. Otherwise the model will
        simply be initialized without the weights.

    Returns
    -------
    nn.Module
        an fms equivalent implementation of an HF model
    """
    get_model_kwargs = _infer_model_configuration(
        model_id_or_path, download_weights=initialize_model_with_weights
    )

    return get_model(
        source="hf",
        device_type=device_type,
        data_type=data_type,
        distributed_strategy=distributed_strategy,
        checkpoint_sharding=checkpoint_sharding,
        group=group,
        **get_model_kwargs,
    )
