from typing import Optional, Tuple
from typing_extensions import Unpack

from fms.modules.attention import SDPAAttentionKwargs
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.hf.gpt_oss.configuration_gpt_oss_hf import HFAdaptedGptOssConfig
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture
from fms.models.gpt_oss import GPTOSS, GPTOSSHeadless

import torch
from torch import nn
from torch.nn import functional as F


class HFAdapterGptOssDecoder(HFDecoder):
    def __init__(self, model: GPTOSSHeadless, config: PretrainedConfig):
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
    
    _tied_weights_keys = ["decoder.model.embedding.weight", "embedding.weight"]
    _keys_to_ignore_on_save = ["embedding.weight"]

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
            model = GPTOSSHeadless(**params)
            decoder = model if decoder is None else decoder
            embedding = model.embedding if embedding is None else embedding

        # these are now huggingface compatible
        decoder = HFAdaptedGptOssHeadless(decoder, config)
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
        Overriding _prepare_inputs_for_generation to include position_ids requirements for llama batch processing
        """
        position_ids = model_kwargs.pop("position_ids", None)

        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1)

        # Add more cached rope freqs if over cached number
        max_expected_len = input_ids.shape[1] + torch.max(position_ids)
        if max_expected_len > self.decoder.model.rot_emb.rope_scaling.orig_max_seq_len:
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

class HFAdaptedGptOssForCausalLM(HFAdaptedGptOssHeadless):
    def __init__(self, config: HFAdaptedGptOssConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    @classmethod
    def _hf_model_from_fms(
        cls, model: GPTOSS, config: HFAdaptedGptOssConfig
    ) -> "HFAdaptedGptOssForCausalLM":
        return cls(
            config=config,
            decoder=model.base_model,
            embedding=model.base_model.embedding,
            lm_head=model.head,
        )