"""
Builder funcs, which consume a transformers PretrainedConfig, and create
a dict of config_params, which should be expanded and passed as overrides
to the model at init time.
"""

# Used in Llava Next for Granite vision
from fms.models.siglip_vision import SiglipVisionConfig
from fms.models.granite import GraniteConfig
from fms.models.llama import LLaMAConfig

from transformers import PretrainedConfig


def build_llama_params(config: PretrainedConfig) -> dict:
    """Param builder for mapping LlamaForCausalLM to FMS."""
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
    return model_params_with_common_opts(
        config, config_params, inner_dim=config.intermediate_size
    )


def build_gpt_bigcode_params(config: PretrainedConfig) -> dict:
    """Param builder for mapping GPTBigCodeForCausalLM to FMS."""
    config_params = {
        "ln_eps": config.layer_norm_epsilon,
        "multiquery_attn": config.multi_query,
        "emb_dim": config.hidden_size,
        "max_expected_seq_len": config.n_positions,
    }
    return model_params_with_common_opts(
        config, config_params, inner_dim=config.n_inner
    )


def build_mixtral_params(config: PretrainedConfig) -> dict:
    """Param builder for mapping MixtralForCausalLM to FMS."""
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


def build_roberta_params(config: PretrainedConfig, is_classify: bool = False) -> dict:
    """Param builder for mapping
        - RobertaForMaskedLM, RobertaForQuestionAnswering when is_classify is False
        - RobertaForSequenceClassification when is_classify is True
    to FMS. In the latter case, this should be wrapped in a partial at registration
    time to override the default value and align with the ParamBuilderFunc signature.
    """
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
    return model_params_with_common_opts(
        config, config_params, inner_dim=config.intermediate_size
    )


def build_granite_params(config: PretrainedConfig) -> dict:
    """Param builder for mapping GraniteForCausalLM to FMS."""
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
    return model_params_with_common_opts(
        config, config_params, inner_dim=config.intermediate_size
    )


def build_granite_moe_hybrid_params(config: PretrainedConfig) -> dict:
    """Param builder for mapping Granite4 / GraniteMoeHybrid model to FMS."""
    # Currently we are configuring granite_moe_hybrid model for the
    # granite-v4 dense version. In future, based on the configuration
    # we may route to different architectures or classes.

    config_params = {
        "attn_bias": getattr(config, "attention_bias", False),
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
    return model_params_with_common_opts(
        config, config_params, inner_dim=config.intermediate_size
    )


def build_mistral_params(config: PretrainedConfig) -> dict:
    """Param builder for mapping MistralForCausalLM to FMS."""
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
    return model_params_with_common_opts(
        config, config_params, inner_dim=config.intermediate_size
    )


def build_bamba_params(config: PretrainedConfig) -> dict:
    """Param builder for mapping BambaForCausalLM to FMS."""
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
    return model_params_with_common_opts(
        config, config_params, inner_dim=config.intermediate_size
    )


def build_siglip_vision_params(config: PretrainedConfig) -> dict:
    """
    Param builder for extracting the Siglip vision encoder for wrapping modules,
    i.e.,
        - LlavaNextForConditionalGeneration (granite vision only)
        - SiglipModel
    to FMS.

    NOTE that this does not consider the text encoder for standalone siglip.
    """
    vision_cfg = config.vision_config

    config_params = {
        "hidden_size": vision_cfg.hidden_size,
        "intermediate_size": vision_cfg.intermediate_size,
        "nlayers": vision_cfg.num_hidden_layers,
        "nheads": vision_cfg.num_attention_heads,
        "num_channels": vision_cfg.num_channels,
        "image_size": vision_cfg.image_size,
        "patch_size": vision_cfg.patch_size,
        "hidden_act": vision_cfg.hidden_act,
        "layer_norm_eps": vision_cfg.layer_norm_eps,
        "attention_dropout": vision_cfg.attention_dropout,
    }
    use_navit = getattr(vision_cfg, "use_navit_position_buckets", None)
    if use_navit is None:
        model_type = getattr(vision_cfg, "model_type", None)
        # Only set to True if model_type matches, not False
        if model_type in ("idefics3", "idefics3_vision"):
            use_navit = True
    if use_navit is None:
        # Fallback heuristic for configs with max_image_size
        use_navit = hasattr(vision_cfg, "max_image_size")
    config_params["use_navit_position_buckets"] = bool(use_navit)
    # Don't build common opts for the vision encoder
    return config_params


def build_llava_next_params(config: PretrainedConfig) -> dict:
    """Param builder for mapping LlavaNextForConditionalGeneration to FMS."""
    config_params = {
        "image_token_index": config.image_token_index,
        "image_grid_pinpoints": config.image_grid_pinpoints,
        "vision_feature_layer": config.vision_feature_layer,
        "vision_feature_select_strategy": (config.vision_feature_select_strategy),
    }

    # TODO abstract and allow recursive config param / model config init
    if config.text_config.model_type != "granite":
        raise ValueError(
            "FMS implementation of LlavaNext currently supports only Granite language model"
        )
    if config.vision_config.model_type != "siglip_vision_model":
        raise ValueError(
            "FMS implementation of LlavaNext currently supports only Siglip vision model"
        )

    vision_config_params = build_siglip_vision_params(config)
    config_params["vision_config"] = SiglipVisionConfig(**vision_config_params)
    text_config_params = build_granite_params(config.text_config)
    config_params["text_config"] = GraniteConfig(**text_config_params)
    # Don't see common opts for the VLM; they'll generally be set in the LLM recursively
    return config_params


def build_idefics3_params(config: PretrainedConfig) -> dict:
    """Param builder for mapping Idefics3ForConditionalGeneration (SmolVLM/Idefics3) to FMS."""
    config_params: dict = {}

    image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is not None:
        config_params["image_token_id"] = int(image_token_id)

    # Vision config (SigLIP)
    vision_config_params = build_siglip_vision_params(config)
    config_params["vision_config"] = SiglipVisionConfig(**vision_config_params)

    # Text config (LLaMA-like)
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is None:
        raise ValueError("Idefics3 config missing required text_config")
    text_config_params = build_llama_params(text_cfg)
    # Idefics3/SmolVLM uses a non-zero pad token; ensure it is propagated.
    if getattr(text_cfg, "pad_token_id", None) is not None:
        text_config_params["pad_id"] = int(text_cfg.pad_token_id)
    config_params["text_config"] = LLaMAConfig(**text_config_params)

    # Connector + packing parameters.
    #
    # SmolVLM/Idefics3 exposes the connector downsample as `pixel_shuffle_factor` on the *text_config*
    # in HF (even though it affects how vision tokens are packed). We intentionally source it from
    # `text_cfg` to match HF behavior.
    pixel_shuffle_factor = getattr(text_cfg, "pixel_shuffle_factor", 4)
    connector_scale = (
        int(pixel_shuffle_factor) if pixel_shuffle_factor is not None else 4
    )
    config_params["connector_scale"] = connector_scale

    image_size = config_params["vision_config"].image_size
    patch_size = config_params["vision_config"].patch_size
    patches_per_side = int(image_size) // int(patch_size)
    config_params["image_span_len"] = (patches_per_side // connector_scale) ** 2

    return config_params


def build_mpnet_params(config: PretrainedConfig) -> dict:
    """Param builder for mapping MPNetForMaskedLM to FMS."""
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
        "relative_attention_num_buckets": (config.relative_attention_num_buckets),
    }
    return model_params_with_common_opts(
        config, config_params, inner_dim=config.intermediate_size
    )


def build_bert_params(config: PretrainedConfig, is_classify: bool = False) -> dict:
    """Param builder for mapping
        - BertForMaskedLM when is_classify is False
        - BertForSequenceClassification when is_classify is True
    to FMS. In the latter case, this should be wrapped in a partial at registration
    time to override the default value and align with the ParamBuilderFunc signature.
    """
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
    return model_params_with_common_opts(
        config, config_params, inner_dim=config.intermediate_size
    )


def model_params_with_common_opts(
    config: PretrainedConfig, config_params: dict, inner_dim: int
) -> dict:
    """
    Adds additional kwargs to the config params that are common
    to most non multimodal architectures.

    Args:
    config: The Transformers PretrainedConfig being mapped.
    config_params: The FMS model kwargs created by the corresponding param builder.
    inner_dim: The internal dimension of the model; this is explicitly passed since
        the key is not standardized across HF config subclasses.
    """
    common_params = {
        "src_vocab_size": config.vocab_size,
        "nheads": config.num_attention_heads,
        "nlayers": config.num_hidden_layers,
        "hidden_grow_factor": inner_dim / config.hidden_size,
        "tie_heads": config.tie_word_embeddings,
    }

    # If we have any overlapping keys with the common params coming
    # from the builder, raise if they have conflicting values.
    overlap = set(common_params.keys()).intersection(set(config_params.keys()))
    mismatches = [
        dup_key
        for dup_key in overlap
        if common_params[dup_key] != config_params[dup_key]
    ]
    if mismatches:
        raise ValueError(
            f"Model param builder uses common params, but has conflicting values for key(s) {mismatches}"
        )

    return {**config_params, **common_params}
