from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.gpt_bigcode import GPTBigCode, GPTBigCodeHeadless
from fms.models.hf.gpt_bigcode.configuration_gpt_bigcode_hf import (
    HFAdaptedGPTBigCodeConfig,
)
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture


class HFAdaptedGPTBigCodeDecoder(HFDecoder):
    """Adapter for the GPTBigCodeDecoder"""

    def __init__(self, model: nn.Module, config: PretrainedConfig):
        super().__init__(model, config, attention_mask_dim=3)

    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if kwargs.get("mask", None) is None:
            kwargs["mask"] = attention_mask

        output, cache = self.model(
            x=input_ids,
            position_ids=position_ids,
            past_key_value_states=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output, past_key_values=cache
        )


class HFAdaptedGPTBigCodeHeadless(HFDecoderModelArchitecture):
    """This is the Adapter for the base gpt_bigcode architecture"""

    # attributes required by HF
    config_class = HFAdaptedGPTBigCodeConfig
    base_model_prefix = "hf_adapted_gpt_bigcode"

    def __init__(
        self,
        config: PretrainedConfig,
        decoder: Optional[GPTBigCodeHeadless] = None,
        embedding: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        # in the case we have not yet received the encoder/decoder/embedding, initialize it here
        if decoder is None or embedding is None:
            params = config.to_dict()
            params["pad_id"] = params.pop("pad_token_id")
            model = GPTBigCode(**params)
            decoder = model.base_model if decoder is None else decoder
            embedding = model.base_model.embedding if embedding is None else embedding

        # these are now huggingface compatible
        decoder = HFAdaptedGPTBigCodeDecoder(decoder, config)
        super().__init__(decoder, embedding, config, *args, **kwargs)


class HFAdaptedGPTBigCodeForCausalLM(
    LMHeadModelLMHeadMixin, HFAdaptedGPTBigCodeHeadless
):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["embedding.weight", "lm_head.weight"]

    def __init__(self, config: HFAdaptedGPTBigCodeConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    def _tie_weights(self):
        # We know that FMS always saves the LM head weight, so ensure the right pointer is shared
        self.embedding.weight = self.lm_head.weight
        self.decoder.model.embedding.weight = self.embedding.weight

    @classmethod
    def _hf_model_from_fms(
        cls, model: nn.Module, config: HFAdaptedGPTBigCodeConfig
    ) -> "HFAdaptedGPTBigCodeForCausalLM":
        return cls(
            config=config,
            decoder=model.base_model,
            embedding=model.base_model.embedding,
            lm_head=model.head,
        )
