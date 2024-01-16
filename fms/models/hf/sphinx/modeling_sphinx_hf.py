from typing import Optional, Tuple

import torch
import torch.nn as nn
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.sphinx import SphinxConfig, Sphinx


class HFAdaptedSphinxConfig(PretrainedConfig):
    model_type = "hf_adapted_sphinx"
    attribute_map = {
        "vocab_size": "src_vocab_size",
        "hidden_size": "emb_dim",
        "num_attention_heads": "nheads",
        "num_hidden_layers": "nlayers",
    }

    def __init__(
        self,
        src_vocab_size: Optional[int] = 32000,
        emb_dim: Optional[int] = 4096,
        norm_eps: float = 1e-6,
        nheads: int = 32,
        kvheads: int = 0,
        nlayers: int = 32,
        # note this is different from the non-hf config (which is -1), hf keeps a different default
        pad_token_id: int = 0,
        hidden_grow_factor: float = 8 / 3,
        multiple_of: int = 256,
        activation_fn: str = "swish",
        p_dropout: float = 0.0,
        max_expected_seq_len: int = 2048,
        use_cache: bool = True,
        eos_token_id: int = 2,
        bos_token_id: int = 1,
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
    def from_fms_config(cls, config: SphinxConfig, **hf_kwargs):
        config_dict = config.as_dict()
        config_dict["pad_token_id"] = config_dict.pop("pad_id")
        return cls.from_dict(config_dict, **hf_kwargs)


class HFAdaptedSphinxDecoder(HFDecoder):
    """Adapter for the sphinx decoder"""

    def __init__(self, model: Sphinx, config: PretrainedConfig):
        super().__init__(model, config, attention_mask_dim=3)

    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        attn_algorithm: Optional[
            str
        ] = None,  # this can be passed in from top most forward
        *args,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        output = self.model._helper(
            x_in=input_ids,
            mask=attention_mask,
            position_ids=position_ids,
            past_key_value_states=past_key_values,
            use_cache=use_cache,
            attn_algorithm=attn_algorithm,
        )

        present_key_values = None
        if isinstance(output, tuple):
            output, present_key_values = output
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output, past_key_values=present_key_values
        )


class HFAdaptedSphinxHeadless(HFDecoderModelArchitecture):
    """This is the Adapter for the base sphinx architecture"""

    # attributes required by HF
    config_class = HFAdaptedSphinxConfig
    base_model_prefix = "hf_adapted_sphinx"

    def __init__(
        self,
        config: PretrainedConfig,
        decoder: Optional[nn.Module] = None,
        embedding: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):

        # in the case we have not yet received the encoder/decoder/embedding, initialize it here
        if decoder is None or embedding is None:
            params = config.to_dict()
            model = Sphinx(pad_id=params.pop("pad_token_id"), **params)
            decoder = model if decoder is None else decoder
            embedding = model.shared.emb if embedding is None else embedding

        # these are now huggingface compatible
        decoder = HFAdaptedSphinxDecoder(decoder, config)
        super().__init__(decoder, embedding, config, *args, **kwargs)

    def _prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs,
    ) -> dict:
        """
        Overriding _prepare_inputs_for_generation to include position_ids requirements for sphinx batch processing
        """
        position_ids = model_kwargs.pop("position_ids", None)

        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1)

        # Add more cached rope freqs if over cached number
        max_expected_len = input_ids.shape[1] + torch.max(position_ids)
        if max_expected_len > self.decoder.model.rot_emb.max_seq_len:
            self.decoder.model.rot_emb.compute_freqs_cis(
                input_ids.device, max_expected_len
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": position_ids,
            **model_kwargs,
        }


class HFAdaptedSphinxForCausalLM(LMHeadModelLMHeadMixin, HFAdaptedSphinxHeadless):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["embedding.weight", "lm_head.weight"]

    def __init__(self, config: HFAdaptedSphinxConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    @classmethod
    def _hf_model_from_fms(
        cls, model: Sphinx, config: HFAdaptedSphinxConfig
    ) -> "HFAdaptedSphinxForCausalLM":
        return cls(
            config=config,
            decoder=model,
            embedding=model.shared.emb,
            lm_head=model.shared.head,
        )

    # overriding this to enable tensor-parallel since it requires a WordEmbedding forward
    # in the future WordEmbedding should be split up
    def _lm_head(self, input_ids, *args, **kwargs):
        return self.decoder.model.shared(input_ids, reverse=True)
