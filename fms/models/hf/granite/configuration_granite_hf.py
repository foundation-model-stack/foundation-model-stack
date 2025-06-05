from typing import Optional

from transformers import PretrainedConfig

from fms.models.granite import GraniteConfig


class HFAdaptedGraniteConfig(PretrainedConfig):
    model_type = "hf_adapted_granite"
    attribute_map = {
        "vocab_size": "src_vocab_size",
        "hidden_size": "emb_dim",
        "num_attention_heads": "nheads",
        "num_hidden_layers": "nlayers",
        "num_key_value_heads": "kvheads",
        "rms_norm_eps": "norm_eps",
        "max_position_embeddings": "max_expected_seq_len",
        "rope_theta": "rope_theta",
        "attention_dropout": "p_dropout",
    }

    attention_dropout = 0.0

    def __init__(
        self,
        src_vocab_size: Optional[int] = 49155,
        emb_dim: Optional[int] = 2048,
        hidden_dim: Optional[int] = 50,
        nlayers: int = 40,
        nheads: int = 32,
        kvheads: int = 8,
        pad_token_id: int = 0,
        max_expected_seq_len: int = 4096,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        p_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        use_cache: bool = True,
        is_decoder: bool = True,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.norm_eps = norm_eps
        self.nheads = nheads
        self.kvheads = kvheads
        self.nlayers = nlayers
        self.rope_theta = rope_theta
        self.p_dropout = p_dropout
        self.max_expected_seq_len = max_expected_seq_len
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            is_decoder=is_decoder,
            tie_word_embeddings=kwargs.pop(
                "tie_word_embeddings", False
            ),  # note: This was added here as we handle tying of heads with our underlying model, we may want to revisit this in future
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
    def from_fms_config(cls, config: GraniteConfig, **hf_kwargs):
        config_dict = config.as_dict()
        config_dict["pad_token_id"] = config_dict.pop("pad_id")
        return cls.from_dict(config_dict, **hf_kwargs)
