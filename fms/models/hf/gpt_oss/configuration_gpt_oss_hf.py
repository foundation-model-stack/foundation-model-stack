from typing import Optional

from transformers import PretrainedConfig

from fms.models.gpt_oss import GPTOSSConfig

class HFAdaptedGptOssConfig(PretrainedConfig):
    model_type = "gpt_oss"
    attribute_map = {
        "vocab_size": "src_vocab_size",
        "hidden_size": "dim",
        "num_attention_heads": "nheads",
        "num_hidden_layers": "nlayers",
        "num_key_value_heads": "kvheads",
        "num_local_experts": "num_experts",
        "num_experts_per_tok": "top_k_experts",
        "intermediate_size": "hidden_dim",
        "rms_norm_eps": "norm_eps",
        "max_position_embeddings": "max_expected_seq_len",
        "rope_theta": "rope_base",
        "attention_dropout": "p_dropout",
    }

    def __init__(
        self,
        num_hidden_layers: int = 36,
        num_local_experts: int = 128,
        vocab_size: int = 201088,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        head_dim: int = 64,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        sliding_window: int = 128,
        rope_theta: float = 150000.0,
        tie_word_embeddings=False,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        max_position_embeddings=131072,
        rms_norm_eps: float = 1e-5,
        rope_scaling={"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": False},
        attention_dropout: float = 0.0,
        num_experts_per_tok=4,
        router_aux_loss_coef: float = 0.9,
        output_router_logits=False,
        use_cache=True,
        layer_types=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_local_experts = num_local_experts
        self.sliding_window = sliding_window
        self.num_experts_per_tok = num_experts_per_tok
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        self.attention_bias = True
        self.max_position_embeddings = max_position_embeddings
        self.router_aux_loss_coef = router_aux_loss_coef
        self.output_router_logits = output_router_logits
        self.use_cache = use_cache
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
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
def from_fms_config(cls, config: GPTOSSConfig, **hf_kwargs):
    config_dict = config.as_dict()
    return cls.from_dict(config_dict, **hf_kwargs)