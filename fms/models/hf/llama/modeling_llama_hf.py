from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.hf.llama.configuration_llama_hf import HFAdaptedLLaMAConfig
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture
from fms.models.llama import LLaMA


class HFAdaptedLLaMADecoder(HFDecoder):
    """Adapter for the LLaMA decoder"""

    def __init__(self, model: LLaMA, config: PretrainedConfig):
        super().__init__(model, config, attention_mask_dim=3)

    def set_input_embeddings(self, value: nn.Module):
        self.model.shared.emb = value

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


class HFAdaptedLLaMAHeadless(HFDecoderModelArchitecture):
    """This is the Adapter for the base granite architecture"""

    # attributes required by HF
    config_class = HFAdaptedLLaMAConfig
    base_model_prefix = "hf_adapted_llama"

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
            model = LLaMA(pad_id=params.pop("pad_token_id"), **params)
            decoder = model if decoder is None else decoder
            embedding = model.shared.emb if embedding is None else embedding

        # these are now huggingface compatible
        decoder = HFAdaptedLLaMADecoder(decoder, config)
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


class HFAdaptedLLaMAForCausalLM(LMHeadModelLMHeadMixin, HFAdaptedLLaMAHeadless):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["embedding.weight", "lm_head.weight"]

    def __init__(self, config: HFAdaptedLLaMAConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    @classmethod
    def _hf_model_from_fms(
        cls, model: LLaMA, config: HFAdaptedLLaMAConfig
    ) -> "HFAdaptedLLaMAForCausalLM":
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
