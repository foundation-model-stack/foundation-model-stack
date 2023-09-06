from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.hf.llama.configuration_llama_hf import LLaMAHFConfig
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture
from fms.models.llama import LLaMA

class LLaMAHFDecoder(HFDecoder):
    """Adapter for the LlamaDecoder"""

    def __init__(self, model: LLaMA, config: PretrainedConfig):
        super().__init__(model, config, attention_mask_dim=3)

    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        attn_algorithm: Optional[str] = None,  # this can be passed in from top most forward
        *args,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        output = self.model(
            input_ids,
            attention_mask,
            past_key_value_states=past_key_values,
            use_cache=use_cache,
            attn_algorithm=attn_algorithm,
        )

        present_key_values = None
        if isinstance(output, tuple):
            output, present_key_values = output

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=output, past_key_values=present_key_values)


class LLaMAHF(HFDecoderModelArchitecture):
    """This is the Adapter for the base granite architecture"""

    # attributes required by HF
    config_class = LLaMAHFConfig
    base_model_prefix = "llama_hf"

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
            decoder = model.stack if decoder is None else decoder
            embedding = model.shared.emb if embedding is None else embedding

        # these are now huggingface compatible
        decoder = LLaMAHFDecoder(decoder, config)
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
        Overriding _prepare_inputs_for_generation to include start_pos requirements for llama batch processing
        """
        # this will find the index of the first token to attend to in each sequence in the batch and ignore the rest
        # this is following huggingface llama position_ids param
        # there are 3 scenarios:
        #      (1) if start_pos (position_ids from hf) is provided, we will use this directly
        #      (2) if attention mask is provided, we will create the start_pos based on the attention_mask first token to attend to for each sequence in the batch
        #      (3) if neither start_pos or attention_mask are provided, no start_pos will be set to the model and we will just assume the start_pos is the beginning
        # considering the position_ids name here as it is more standard for hf
        position_ids = model_kwargs.get("position_ids", None)
        if position_ids is not None:
            start_pos = (position_ids == 0).max(1).indices.unsqueeze(1)
        elif attention_mask is not None:
            start_pos = attention_mask.long().cumsum(-1) - 1
            start_pos = (start_pos == 0).max(1).indices.unsqueeze(1)
        else:
            start_pos = None

        # When you are using dynamic batching with left-padding + kv-cache,
        # the current index in the sentence needs to be computed by
        # subtracting the current token index in the batch minus
        # the original starting positions
        use_cache_loc = self.config.use_cache if use_cache is None else use_cache
        if use_cache_loc and past_key_values is not None and isinstance(start_pos, torch.Tensor):
            start_pos = start_pos - past_key_values[0][0].shape[2]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "start_pos": start_pos,
            **model_kwargs,
        }


class LLaMAHFForCausalLM(LMHeadModelLMHeadMixin, LLaMAHF):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _tied_weights_keys = ["embedding.weight", "lm_head.weight"]

    def __init__(self, config: LLaMAHFConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    @classmethod
    def _hf_model_from_fms(cls, model: LLaMA, config: LLaMAHFConfig) -> "LLaMAHFForCausalLM":
        return cls(
            config=config,
            decoder=model.stack,
            embedding=model.shared.emb,
            lm_head=model.shared.head,
        )
