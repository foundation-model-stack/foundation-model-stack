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

from packaging.version import Version
from transformers import __version__ as tf_version


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
    ## Address transformers API changes
    if Version(tf_version) >= Version("5.0.0"):
        _keys_to_ignore_on_load_missing = [
            r"decoder\.model\.embedding\.weight",
        ]
        _tied_weights_keys = {
            "decoder.model.embedding.weight": "embedding.weight",
            "lm_head.weight": "embedding.weight",
        }
    else:
        _keys_to_ignore_on_load_missing = [
            r"decoder\.model\.embedding\.weight",
            r"lm_head\.weight",
        ]
        # For transformers < 5.0.0, set to empty list to disable automatic tying
        # We'll handle tying manually in load_state_dict
        _tied_weights_keys = []

    def __init__(self, config: HFAdaptedGPTBigCodeConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """Override to exclude tied weights from state_dict.

        This prevents saving duplicate embeddings. The tied weights will be
        restored during load via load_state_dict().
        """
        state_dict = super().state_dict(*args, **kwargs)
        # Remove decoder.model.embedding.weight as it's tied to embedding.weight
        if "decoder.model.embedding.weight" in state_dict:
            del state_dict["decoder.model.embedding.weight"]
        # For transformers < 5.0.0, also remove lm_head.weight if tied
        if Version(tf_version) < Version("5.0.0") and self.config.tie_word_embeddings:
            if "lm_head.weight" in state_dict:
                del state_dict["lm_head.weight"]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to handle missing decoder.model.embedding.weight and lm_head.weight, and ensure weights are tied after loading"""
        # If decoder.model.embedding.weight and lm_head.weight are missing from state_dict, that's expected
        # because we exclude them in state_dict(). They will be tied to embedding.weight.
        # So we load with strict=False first
        result = super().load_state_dict(state_dict, strict=False, assign=assign)

        # Filter out the expected missing keys from the result
        expected_missing = ["decoder.model.embedding.weight"]
        if self.config.tie_word_embeddings and Version(tf_version) < Version("5.0.0"):
            # For transformers < 5.0.0, lm_head.weight is also excluded from state_dict
            expected_missing.append("lm_head.weight")

        filtered_missing_keys = [
            k for k in result.missing_keys if k not in expected_missing
        ]

        # Manually tie the weights after loading
        if self.config.tie_word_embeddings:
            # Tie decoder.model.embedding to embedding
            if (
                hasattr(self, "decoder")
                and hasattr(self.decoder, "model")
                and hasattr(self.decoder.model, "embedding")
            ):
                self.decoder.model.embedding.weight = self.embedding.weight
            # Tie lm_head to embedding
            if hasattr(self, "lm_head") and hasattr(self.lm_head, "weight"):
                self.lm_head.weight = self.embedding.weight

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

        return result

    @classmethod
    def _hf_model_from_fms(
        cls, model: nn.Module, config: HFAdaptedGPTBigCodeConfig
    ) -> "HFAdaptedGPTBigCodeForCausalLM":
        # Respect the FMS model's tie_heads setting
        config.tie_word_embeddings = model.config.tie_heads
        return cls(
            config=config,
            decoder=model.base_model,
            embedding=model.base_model.embedding,
            lm_head=model.head,
        )
