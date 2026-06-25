"""GraniteSWA model configuration for transformers AutoConfig.

This is a self-contained ``PretrainedConfig`` subclass that lets
``transformers.AutoConfig.from_pretrained`` recognize ``model_type ==
"granite_swa"`` without requiring a ``granite_swa`` implementation in
transformers itself.

It mirrors the upstream GraniteSWA HF config (Granite 4.5 / sliding-window
attention with learnable attention sinks). FMS does *not* consume this class
directly to build the model: the generic ``hf_pretrained`` loading path reads
the raw config attributes via
``fms.models.hf.config_utils.param_builders.build_granite_swa_params`` and
builds the native FMS ``GraniteSWA`` model. This config only needs to surface
those attributes (populated from ``config.json``), so no ``to_fms_config()`` is
required.
"""

from transformers import PretrainedConfig


class GraniteSWAConfig(PretrainedConfig):
    r"""
    Configuration class for the GraniteSWA model with Sliding Window Attention
    and learnable attention sinks.
    """

    model_type = "granite_swa"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise_allreduce",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise_allreduce",
    }

    def __init__(
        self,
        vocab_size=100352,
        hidden_size=2560,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_attention_heads=20,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=100257,
        eos_token_id=100257,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        embedding_multiplier=1.0,
        logits_scaling=1.0,
        residual_multiplier=1.0,
        attention_multiplier=1.0,
        sliding_window=128,
        layer_types=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.embedding_multiplier = embedding_multiplier
        self.logits_scaling = logits_scaling
        self.residual_multiplier = residual_multiplier
        self.attention_multiplier = attention_multiplier
        self.sliding_window = sliding_window

        if layer_types is None:
            # full attention every 4th layer, sliding window elsewhere
            layer_types = [
                "full_attention" if i % 4 == 0 else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["GraniteSWAConfig"]
