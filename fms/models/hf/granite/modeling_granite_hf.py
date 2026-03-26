from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.hf.granite.configuration_granite_hf import HFAdaptedGraniteConfig
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture
from fms.models.granite import Granite, GraniteHeadless

from packaging.version import Version
from transformers import __version__ as tf_version


class HFAdaptedGraniteDecoder(HFDecoder):
    """Adapter for the Granite decoder"""

    def __init__(self, model: Granite, config: PretrainedConfig):
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

        output = self.model(
            x_in=input_ids,
            position_ids=position_ids,
            past_key_value_states=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        present_key_values = None
        if isinstance(output, tuple):
            output, present_key_values = output
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=output, past_key_values=present_key_values
        )


class HFAdaptedGraniteHeadless(HFDecoderModelArchitecture):
    """This is the Adapter for the base granite architecture"""

    # attributes required by HF
    config_class = HFAdaptedGraniteConfig
    base_model_prefix = "hf_adapted_granite"

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
            model = GraniteHeadless(**params)
            decoder = model if decoder is None else decoder
            embedding = model.embedding if embedding is None else embedding

        # these are now huggingface compatible
        decoder = HFAdaptedGraniteDecoder(decoder, config)
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
        Overriding _prepare_inputs_for_generation to include position_ids requirements for granite batch processing
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


class HFAdaptedGraniteForCausalLM(LMHeadModelLMHeadMixin, HFAdaptedGraniteHeadless):
    ## Address transformers API changes
    if Version(tf_version) >= Version("5.0.0"):
        _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
        _tied_weights_keys = {
            "lm_head.weight": "decoder.model.embedding.weight",
            "embedding.weight": "decoder.model.embedding.weight",
        }
    else:
        _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
        _tied_weights_keys = ["embedding.weight", "lm_head.weight"]

    def __init__(self, config: HFAdaptedGraniteConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Override to exclude decoder.model.embedding.weight from state_dict.

        This prevents saving duplicate embeddings. The decoder's embedding will be
        tied to the main embedding during load via tie_weights().
        """
        state_dict = super().state_dict(*args, **kwargs)
        # Remove decoder.model.embedding.weight as it's tied to embedding.weight
        if "decoder.model.embedding.weight" in state_dict:
            del state_dict["decoder.model.embedding.weight"]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to handle missing decoder.model.embedding.weight and ensure weights are tied after loading"""
        # If decoder.model.embedding.weight is missing from state_dict, that's expected
        # because we exclude it in state_dict(). It will be tied to embedding.weight.
        # So we load with strict=False first
        result = super().load_state_dict(state_dict, strict=False, assign=assign)

        # Filter out the expected missing key from the result
        filtered_missing_keys = [
            k for k in result.missing_keys if k != "decoder.model.embedding.weight"
        ]

        # If strict mode was requested and there are still missing/unexpected keys, raise an error
        if strict and (filtered_missing_keys or result.unexpected_keys):
            error_msgs = []
            if result.unexpected_keys:
                error_msgs.append(
                    f"Unexpected key(s) in state_dict: {', '.join(result.unexpected_keys)}"
                )
            if filtered_missing_keys:
                error_msgs.append(
                    f"Missing key(s) in state_dict: {', '.join(filtered_missing_keys)}"
                )
            if error_msgs:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t"
                    + "\n\t".join(error_msgs)
                )

        # Manually tie decoder.model.embedding.weight to embedding.weight after loading
        if self.decoder.model.embedding is not self.embedding:
            self.decoder.model.embedding.weight = self.embedding.weight

        # Only tie lm_head to embedding if tie_word_embeddings is True AND lm_head.weight was not in the state_dict
        # (if lm_head.weight was in state_dict, it means they should be separate)
        if self.config.tie_word_embeddings and "lm_head.weight" not in state_dict:
            self.lm_head.weight = self.embedding.weight

        return result

    @classmethod
    def _hf_model_from_fms(
        cls, model: Granite, config: HFAdaptedGraniteConfig
    ) -> "HFAdaptedGraniteForCausalLM":
        # Set tie_word_embeddings based on the FMS model's tie_heads config
        config.tie_word_embeddings = config.tie_heads
        return cls(
            config=config,
            decoder=model.base_model,
            embedding=model.base_model.embedding,
            lm_head=model.head,
        )
