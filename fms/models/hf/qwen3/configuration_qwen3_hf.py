from typing import Optional

from transformers import PretrainedConfig

from fms.models.qwen3 import Qwen3Config


class HFAdaptedQwen3Config(PretrainedConfig):
    model_type = "hf_adapted_qwen3"
    attribute_map = {
        "vocab_size": "src_vocab_size",
        "hidden_size": "emb_dim",
        "num_attention_heads": "nheads",
        "num_hidden_layers": "nlayers",
    }

    def __init__(
        self,
        src_vocab_size: int = 151_936,
        emb_dim: int = 2560,
        norm_eps: float = 1e-6,
        nheads: int = 32,
        kvheads: int = 8,
        nlayers: int = 36,
        pad_token_id: int = 0,
        hidden_grow_factor: float = 9728 / 2560,
        multiple_of: int = 256,
        activation_fn: str = "swish",
        p_dropout: float = 0.0,
        max_expected_seq_len: int = 40960,
        attn_bias: bool = False,
        mlp_bias: bool = False,
        rope_theta: float = 1000000.0,
        rope_scaling: dict = {},
        head_dim: int = 128,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        eos_token_id: int = 151645,
        bos_token_id: int = 151643,
        is_decoder: bool = True,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.emb_dim = emb_dim
        self.norm_eps = norm_eps
        self.nheads = nheads
        self.kvheads = kvheads
        self.nlayers = nlayers
        self.hidden_grow_factor = hidden_grow_factor
        self.multiple_of = multiple_of
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.max_expected_seq_len = max_expected_seq_len
        self.attn_bias = attn_bias
        self.mlp_bias = mlp_bias
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            is_decoder=is_decoder,
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
    def from_fms_config(cls, config: Qwen3Config, **hf_kwargs):
        config_dict = config.as_dict()
        config_dict["pad_token_id"] = config_dict.pop("pad_id")
        config_dict["tie_word_embeddings"] = config_dict.pop("tie_heads", True)
        return cls.from_dict(config_dict, **hf_kwargs)
