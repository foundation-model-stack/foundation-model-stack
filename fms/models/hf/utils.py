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
    from fms.models.hf import (
        _causal_lm_models,
        _headless_models,
        _masked_lm_models,
    )

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
        needs_correction.unsqueeze(1),
        mask_encoder[:, 0].unsqueeze(1),
        mask_decoder,
    )

    return mask_encoder.unsqueeze(1) == mask_decoder.unsqueeze(2)


def _map_model_config(architecture, config):
    # Map HF model config to FMS model config
    infer_common_params = True
    config_params = {}
    if architecture == "LlamaForCausalLM":
        architecture = "llama"
        config_params = build_llama_params(config)

    elif architecture == "GPTBigCodeForCausalLM":
        architecture = "gpt_bigcode"
        config_params = build_gpt_bigcode_params(config)

    elif architecture == "MixtralForCausalLM":
        architecture = "mixtral"
        config_params = build_mixtral_params(config)

    elif architecture == "RobertaForMaskedLM":
        architecture = "roberta"
        config_params = build_roberta_params(config, is_classify=False)

    elif architecture == "RobertaForQuestionAnswering":
        architecture = "roberta_question_answering"
        config_params = build_roberta_params(config, is_classify=False)

    elif architecture == "RobertaForSequenceClassification":
        architecture = "roberta_classification"
        config_params = build_roberta_params(config, is_classify=True)

    elif architecture == "GraniteForCausalLM":
        architecture = "granite"
        config_params = build_granite_params(config)

    elif architecture == "MistralForCausalLM":
        architecture = "mistral"
        config_params = build_mistral_params(config)

    elif architecture == "BambaForCausalLM":
        architecture = "bamba"
        config_params = build_bamba_params(config)


    elif architecture == "SiglipModel":
        architecture = "siglip_vision"
        # For siglip, we only use the vision encoder
        config_params = build_siglip_vision_params(config)

    # Granite vision
    elif architecture == "LlavaNextForConditionalGeneration":
        architecture = "llava_next"
        config_params = build_llava_next_params(config)

    elif architecture == "MPNetForMaskedLM":
        architecture = "mpnet"
        config_params = build_mpnet_params(config)

    elif architecture == "BertForMaskedLM":
        architecture = "bert"
        config_params = build_bert_params(config, is_classify=False)

    elif architecture == "BertForSequenceClassification":
        architecture = "bert_classification"
        config_params = build_bert_params(config, is_classify=True)

    else:
        raise ValueError(
            "FMS model implementations currently only support LlamaForCausalLM, GPTBigCodeForCausalLM, MixtralForCausalLM, RobertaForMaskedLM, RobertaForQuestionAnswering, RobertaForSequenceClassification, GraniteForCausalLM, MistralForCausalLM, BambaForCausalLM, SiglipModel, LlavaNextForConditionalGeneration, MPNetForMaskedLM, BertForMaskedLM, and BertForSequenceClassification"
        )
    return architecture, config_params

### Config builders for different model architectures
def build_llama_params(config):
    config_params = {
        "attn_bias": getattr(config, "attention_bias", False),
        "mlp_bias": getattr(config, "mlp_bias", False),
        "kvheads": config.num_key_value_heads,
        "norm_eps": config.rms_norm_eps,
        "multiple_of": 1,
        "emb_dim": config.hidden_size,
        "max_expected_seq_len": config.max_position_embeddings,
    }
    # New in Llama 3
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is not None:
        config_params["rope_theta"] = rope_theta
    # New in Llama 3.1
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        config_params["rope_scaling"] = rope_scaling
    
    # Apply the common params
    return model_params_with_common_opts(config, config_params, inner_dim=config.intermediate_size)

def build_gpt_bigcode_params(config):
    config_params = {
        "ln_eps": config.layer_norm_epsilon,
        "multiquery_attn": config.multi_query,
        "emb_dim": config.hidden_size,
        "max_expected_seq_len": config.n_positions,
    }
    return model_params_with_common_opts(config, config_params, inner_dim=config.n_inner)

def build_mixtral_params(config):
    inner_dim = config.intermediate_size
    config_params = {
        "dim": config.hidden_size,
        "hidden_dim": inner_dim,
        "norm_eps": config.rms_norm_eps,
        "kv_heads": config.num_key_value_heads,
        "num_experts": config.num_local_experts,
        "top_k_experts": config.num_experts_per_tok,
        "rope_base": config.rope_theta,
        "max_expected_seq_len": config.max_position_embeddings,
    }
    return model_params_with_common_opts(config, config_params, inner_dim=inner_dim)

def build_roberta_params(config, is_classify: bool):
    config_params = {
        "emb_dim": config.hidden_size,
        "pad_id": config.pad_token_id,
        "max_pos": config.max_position_embeddings - 2,
        "p_dropout": config.hidden_dropout_prob,
        "norm_eps": config.layer_norm_eps,
        "activation_fn": config.hidden_act,
        "type_vocab_size": config.type_vocab_size,
        "pos_emb": "roberta",
    }

    if is_classify:
        # The only difference for classify is num_classes
        config_params["num_classes"] = config.num_labels
    return model_params_with_common_opts(config, config_params, inner_dim=config.intermediate_size)


def build_granite_params(config):
    config_params = {
        "attn_bias": getattr(config, "attention_bias", False),
        "mlp_bias": getattr(config, "mlp_bias", False),
        "kvheads": config.num_key_value_heads,
        "norm_eps": config.rms_norm_eps,
        "multiple_of": 1,
        "emb_dim": config.hidden_size,
        "max_expected_seq_len": config.max_position_embeddings,
        "residual_multiplier": config.residual_multiplier,
        "attention_multiplier": config.attention_multiplier,
        "logits_scaling": config.logits_scaling,
        "embedding_multiplier": config.embedding_multiplier,
        "rope_theta": config.rope_theta,
        "activation_fn": config.hidden_act,
        "head_dim": getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        ),
    }
    return model_params_with_common_opts(config, config_params, inner_dim=config.intermediate_size)

def build_mistral_params(config):
    config_params = {
        "activation_fn": config.hidden_act,
        "emb_dim": config.hidden_size,
        "max_expected_seq_len": config.max_position_embeddings,
        "kvheads": config.num_key_value_heads,
        "p_dropout": config.attention_dropout,
        "head_dim": (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        ),
        "norm_eps": config.rms_norm_eps,
        "rope_base": config.rope_theta,
        "sliding_window": config.sliding_window,
    }
    return model_params_with_common_opts(config, config_params, inner_dim=config.intermediate_size)

def build_bamba_params(config):
    config_params = {
        "kvheads": config.num_key_value_heads,
        "p_dropout": config.attention_dropout,
        "activation_fn": config.hidden_act,
        "emb_dim": config.hidden_size,
        "chunk_size": config.mamba_chunk_size,
        "use_conv_bias": config.mamba_conv_bias,
        "conv_kernel": config.mamba_d_conv,
        "head_dim": config.mamba_d_head,
        "state_size": config.mamba_d_state,
        "mamba_expand": config.mamba_expand,
        "n_groups": config.mamba_n_groups,
        "mamba_n_heads": config.mamba_n_heads,
        "use_bias": config.mamba_proj_bias,
        "norm_eps": config.rms_norm_eps,
    }
    return model_params_with_common_opts(config, config_params, inner_dim=config.intermediate_size)

def build_siglip_vision_params(config):
    config = config.vision_config # vision encoder only
    config_params = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "nlayers": config.num_hidden_layers,
        "nheads": config.num_attention_heads,
        "num_channels": config.num_channels,
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "hidden_act": config.hidden_act,
        "layer_norm_eps": config.layer_norm_eps,
        "attention_dropout": config.attention_dropout,
    }
    # Don't build common opts for the vision encoder
    return config_params

def build_llava_next_params(config):
    # TODO - make this generic
    from fms.models.siglip_vision import SiglipVisionConfig
    from fms.models.granite import GraniteConfig
    config_params = {
        "image_token_index": config.image_token_index,
        "image_grid_pinpoints": config.image_grid_pinpoints,
        "vision_feature_layer": config.vision_feature_layer,
        "vision_feature_select_strategy": (
            config.vision_feature_select_strategy
        ),
    }

    # TODO make this a warning and abstract the visual encoder / LLM implementations
    if config.text_config.model_type != "granite":
        raise ValueError(
            "FMS implementation of LlavaNext currently supports only Granite language model"
        )
    if config.vision_config.model_type != "siglip_vision_model":
        raise ValueError(
            "FMS implementation of LlavaNext currently supports only Siglip vision model"
        )

    _, vision_config_params = _map_model_config("SiglipModel", config)
    config_params["vision_config"] = SiglipVisionConfig(**vision_config_params)
    _, text_config_params = _map_model_config(
        "GraniteForCausalLM", config.text_config
    )
    config_params["text_config"] = GraniteConfig(**text_config_params)
    # Don't see common opts for the VLM; they'll generally be set in the LLM recursively
    return config_params

def build_mpnet_params(config):
    config_params = {
        "p_dropout": config.attention_probs_dropout_prob,
        "hidden_dropout_prob": config.hidden_dropout_prob,
        "layer_norm_eps": config.layer_norm_eps,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "activation_fn": config.hidden_act,
        "emb_dim": config.hidden_size,
        "max_expected_seq_len": config.max_position_embeddings,
        "pad_id": config.pad_token_id,
        "relative_attention_num_buckets": (
            config.relative_attention_num_buckets
        ),
    }
    return model_params_with_common_opts(config, config_params, inner_dim=config.intermediate_size)

def build_bert_params(config, is_classify: bool):
    config_params = {
        "emb_dim": config.hidden_size,
        "pad_id": config.pad_token_id,
        "max_pos": config.max_position_embeddings,
        "p_dropout": config.hidden_dropout_prob,
        "norm_eps": config.layer_norm_eps,
        "activation_fn": config.hidden_act,
        "type_vocab_size": config.type_vocab_size,
        "pos_emb": "bert",
    }

    if is_classify:
        # The only difference for classify is num_classes
        config_params["num_classes"] = config.num_labels
    return model_params_with_common_opts(config, config_params, inner_dim=config.intermediate_size)

def model_params_with_common_opts(config, config_params, inner_dim):
    common_params = {
        "src_vocab_size": config.vocab_size,
        "nheads": config.num_attention_heads,
        "nlayers": config.num_hidden_layers, 
        "hidden_grow_factor": inner_dim / config.hidden_size,
        "tie_heads": config.tie_word_embeddings,
    }
    # Should not have overlap
    assert not any(common_params) in config_params
    return {**config_params, **common_params}

def infer_model_configuration(
    model_id_or_path: str | os.PathLike,
    download_weights: bool = True,
) -> Dict[str, Any]:
    # if the path does not exist, download it from huggingface and get the local path
    if not os.path.exists(model_id_or_path):
        from huggingface_hub import snapshot_download  # type: ignore

        # in the case we don't want to download the weights, but just create the model from scratch, we will only allow config.json
        if download_weights:
            allow_patterns = [
                "*config.json",
                "tokenizer*",
                "special_tokens_map.json",
            ]

            # mixtral saves safetensors expert sharded, so we will need their pt checkpoints
            # ideally this should be fixed in the adapter in the future
            ignore_patterns = None
            if isinstance(model_id_or_path, str) and model_id_or_path.startswith(
                "mistralai/Mixtral"
            ):
                ignore_patterns = ["*.safetensors"]
                allow_patterns.append("*.pt")
            elif isinstance(model_id_or_path, str) and model_id_or_path.startswith(
                "mistralai/Mistral"
            ):
                ignore_patterns = ["consolidated.safetensors"]
                allow_patterns.append("*.safetensors*")
            else:
                allow_patterns.append("*.safetensors*")
        else:
            allow_patterns = ["config.json"]
            ignore_patterns = None

        model_path = snapshot_download(
            repo_id=str(model_id_or_path),
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns,
        )
    else:
        model_path = str(model_id_or_path)

    config = AutoConfig.from_pretrained(model_path)
    architecture = config.architectures[0]
    architecture, config_params = _map_model_config(architecture, config)

    # infer get_model params
    config_params["architecture"] = architecture
    config_params["variant"] = list_variants(architecture)[0]
    config_params["model_path"] = model_path if download_weights else None

    ## infer quantization parameters
    quant_config = getattr(config, "quantization_config", None)
    if quant_config is not None:
        try:
            from fms_mo.aiu_addons import _infer_quantization_config  # type: ignore[import-untyped,import-not-found]
        except ImportError:
            raise RuntimeError(
                "You need to install fms-model-optimizer to load quantized models"
            )
        linear_config = _infer_quantization_config(quant_config)
        if linear_config:
            config_params["linear_config"] = linear_config

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
