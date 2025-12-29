from fms.models.gpt_oss import GptOssConfig

from transformers import PretrainedConfig


class HFAdaptedGptOssConfig(PretrainedConfig):
    model_type = "hf_adapted_gpt_oss"
    attribute_map = {
        "vocab_size": "src_vocab_size",
        "num_attention_heads": "nheads",
        "num_hidden_layers": "nlayers",
        "num_key_value_heads": "kvheads",
        "num_local_experts": "num_experts",
        "num_experts_per_tok": "top_k_experts",
        "intermediate_size": "hidden_dim",
        "rms_norm_eps": "norm_eps",
        "max_position_embeddings": "max_expected_seq_len",
        "attention_dropout": "p_dropout",
        "attention_bias": "attn_bias",
        "eos_token_id": "eos_token_id",
        "bos_token_id": "bos_token_id",
        "hidden_act": "activation_fn",
        "hidden_size": "emb_dim",
        "sliding_window": "sliding_window",
        "rope_theta": "rope_theta",
        "rope_scaling": "rope_scaling",
        "swiglu_limit": "swiglu_limit",
        "layer_types": "layer_types",
    }

    def __init__(
        self,
        num_experts: int = 32,
        src_vocab_size: int = 201088,
        emb_dim: int = 2880,
        hidden_dim: int = 2880,
        head_dim: int = 64,
        sliding_window: int = 128,
        rope_base: float = 150000.0,
        tie_heads=False,
        activation_fn: str = "silu",
        initializer_range: float = 0.02,
        max_expected_seq_len=131072,
        top_k_experts=4,
        router_aux_loss_coef: float = 0.9,
        output_router_logits=False,
        use_cache=True,
        layer_types=None,
        pad_id=199999,
        nheads: int = 64,
        nlayers: int = 24,
        norm_eps: float = 1e-05,
        kvheads: int = 8,
        p_dropout: float = 0.0,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.emb_dim = emb_dim
        self.nlayers = nlayers
        self.nheads = nheads
        self.num_experts = num_experts
        self.sliding_window = sliding_window
        self.top_k_experts = top_k_experts
        # for backward compatibility
        self.kvheads = kvheads
        self.activation_fn = activation_fn
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.rope_base = rope_base
        self.p_dropout = p_dropout
        self.dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        self.rope_scaling = kwargs.get("rope_scaling")
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        self.attention_bias = True
        self.max_expected_seq_len = max_expected_seq_len
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.use_cache = use_cache
        self.tie_word_embeddings = False
        super().__init__(
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_fms_config(cls, config: GptOssConfig, **hf_kwargs):
        config_dict = config.as_dict()
        return cls.from_dict(config_dict, **hf_kwargs)
