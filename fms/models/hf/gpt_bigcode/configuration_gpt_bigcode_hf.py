from typing import Optional

from transformers import PretrainedConfig

from fms.models.gpt_bigcode import GPTBigCodeConfig


class GPTBigCodeHFConfig(PretrainedConfig):
    model_type = "gpt_bigcode_hf"

    attribute_map = {
        "vocab_size": "src_vocab_size",
        "hidden_size": "emb_dim",
        "num_attention_heads": "nheads",
        "num_hidden_layers": "nlayers",
    }

    def __init__(
        self,
        src_vocab_size: Optional[int] = 49280,
        emb_dim: Optional[int] = 2048,
        emb_kq: Optional[int] = None,
        emb_v: Optional[int] = None,
        nheads: int = 12,
        kvheads: int = 0,
        nlayers: int = 12,
        pad_token_id: int = 0,
        max_pos: int = 512,
        vocab_bias: bool = False,
        use_bias: bool = True,
        hidden_grow_factor: float = 4.0,
        activation_fn: str = "gelu-tanh",
        p_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        ln_eps: float = 1e-5,
        use_cache: bool = True,
        eos_token_id: int = 49152,
        bos_token_id: int = 49152,
        is_decoder: bool = True,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.emb_dim = emb_dim
        self.emb_kq = emb_kq
        self.emb_v = emb_v
        self.nheads = nheads
        self.kvheads = kvheads
        self.nlayers = nlayers
        self.max_pos = max_pos
        self.hidden_grow_factor = hidden_grow_factor
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.emb_dropout = emb_dropout
        self.use_bias = use_bias
        self.vocab_bias = vocab_bias
        self.ln_eps = ln_eps
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            is_decoder=is_decoder,
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
            # note: This was added here as we handle tying of heads with our underlying model, we may want to revisit this in future
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
    def from_fms_config(cls, config: GPTBigCodeConfig, **hf_kwargs):
        config_dict = config.as_dict()
        config_dict["pad_token_id"] = config_dict.pop("pad_id")
        return cls.from_dict(config_dict, **hf_kwargs)
