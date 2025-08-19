from typing import Optional, Tuple
from typing_extensions import Unpack

from fms.modules.attention import SDPAAttentionKwargs
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.hf.gpt_oss.configuration_gpt_oss_hf import HFAdaptedGptOssConfig
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture
from fms.models.gpt_oss import GptOss, GptOssHeadless

import torch
from torch import nn
from torch.nn import functional as F


class HFAdapterGptOssDecoder(HFDecoder):
    def __init__(self, model: GptOss, config: PretrainedConfig):
        super().__init__(model, config, attention_mask_dim=3)

    def adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        **kwargs: Unpack[SDPAAttentionKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if kwargs.get("mask", None) is None:
            kwargs["mask"] = attention_mask

        output = self.model(
            x=input_ids,
            position_ids=position_ids,
            past_key_value_states=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )

        present_key_value = None
        if isinstance(output, tuple):
            output, present_key_value = output
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output, past_key_values=present_key_value
        )


class HFAdaptedGptOssHeadless(HFDecoderModelArchitecture):
    config_class: HFAdaptedGptOssConfig
    base_model_prefix = "hf_adapted_gpt_oss"
    

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
            params["pad_id"] = params.pop("pad_token_id")
            model = GptOssHeadless(**params)
            decoder = model.base_model if decoder is None else decoder
            embedding = model.base_model.embedding if embedding is None else embedding


        # these are now huggingface compatible
        decoder = HFAdaptedGptOssHeadless(decoder, config)
        super().__init__(decoder, embedding, config, *args, **kwargs)


class HFAdaptedGptOssForCausalLM(
    LMHeadModelLMHeadMixin, HFAdaptedGptOssHeadless
    ):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["embedding.weight", "lm_head.weight"]
    def __init__(self, config: HFAdaptedGptOssConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)
    def _tie_weights(self):
        # We know that FMS always saves the LM head weight, so ensure the right pointer is shared
        self.embedding.weight = self.lm_head.weight
        self.decoder.model.embedding.weight = self.embedding.weight

    @classmethod
    def _hf_model_from_fms(
        cls, model: nn.Module, config: HFAdaptedGptOssConfig
    ) -> "HFAdaptedGptOssForCausalLM":
        return cls(
            config=config,
            decoder=model.base_model,
            embedding=model.base_model.embedding,
            lm_head=model.head,
        )
