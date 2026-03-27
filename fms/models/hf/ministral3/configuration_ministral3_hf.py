from transformers import PretrainedConfig


class HFAdaptedMinistral3Config(PretrainedConfig):
    """
    Configuration class for HF-adapted Ministral3 model.

    This config wraps the FMS Ministral3 configuration to make it compatible
    with HuggingFace's AutoModel system.
    """

    model_type = "ministral3"

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=5120,
        intermediate_size=16384,
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=262144,
        rms_norm_eps=1e-5,
        sliding_window=4000,
        attention_dropout=0.0,
        pad_token_id=-1,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_parameters=None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

    def to_fms_config(self):
        """Convert to FMS Ministral3TextConfig"""
        from fms.models.ministral3 import Ministral3TextConfig


        return Ministral3TextConfig(
            src_vocab_size=self.vocab_size,
            emb_dim=self.hidden_size,
            nheads=self.num_attention_heads,
            nlayers=self.num_hidden_layers,
            kvheads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_expected_seq_len=self.max_position_embeddings,
            norm_eps=self.rms_norm_eps,
            sliding_window=self.sliding_window,
            hidden_grow_factor=self.intermediate_size / self.hidden_size,
            pad_id=self.pad_token_id,
            rope_parameters=self.rope_parameters,
        )

