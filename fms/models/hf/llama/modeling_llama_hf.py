from typing import Optional, Tuple
from typing_extensions import Unpack

from fms.modules.attention import SDPAAttentionKwargs
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.hf.llama.configuration_llama_hf import HFAdaptedLLaMAConfig
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture
from fms.models.llama import LLaMA, LLaMAHeadless
from fms.modules.head import LinearClassificationHead

from packaging.version import Version
from transformers import __version__ as tf_version


class HFAdaptedLLaMADecoder(HFDecoder):
    """Adapter for the LLaMA decoder"""

    def __init__(self, model: LLaMA, config: PretrainedConfig):
        super().__init__(model, config, attention_mask_dim=3)

    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        *args,
        **kwargs: Unpack[SDPAAttentionKwargs],
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


class HFAdaptedLLaMAHeadless(HFDecoderModelArchitecture):
    """This is the Adapter for the base llama architecture"""

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
            model = LLaMAHeadless(pad_id=params.pop("pad_token_id"), **params)
            decoder = model if decoder is None else decoder
            embedding = model.embedding if embedding is None else embedding

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


class HFAdaptedLLaMAForCausalLM(LMHeadModelLMHeadMixin, HFAdaptedLLaMAHeadless):
    ## Address transformers API changes
    if Version(tf_version) >= Version("5.0.0"):
        _keys_to_ignore_on_load_missing = [
            r"lm_head.weight",
            r"decoder\.model\.embedding\.weight",
        ]
        _tied_weights_keys = {
            "lm_head.weight": "embedding.weight",
            "decoder.model.embedding.weight": "embedding.weight",
        }
    else:
        _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
        _tied_weights_keys = ["embedding.weight", "lm_head.weight"]

    def __init__(self, config: HFAdaptedLLaMAConfig, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

    def _get_empty_lm_head(self, bias: bool) -> nn.Module:
        """Override to use LinearClassificationHead instead of nn.Linear"""
        return LinearClassificationHead(
            self.config.hidden_size, self.config.vocab_size, bias=bias
        )

    def set_output_embeddings(self, new_embeddings):
        """Override to ensure we always use LinearClassificationHead"""
        if new_embeddings is not None and not isinstance(
            new_embeddings, LinearClassificationHead
        ):
            # If transformers tries to set a regular nn.Linear, convert it to LinearClassificationHead
            if isinstance(new_embeddings, nn.Linear):
                lm_head = LinearClassificationHead(
                    new_embeddings.in_features,
                    new_embeddings.out_features,
                    bias=new_embeddings.bias is not None,
                )
                # Copy the weights and bias
                lm_head.weight = new_embeddings.weight
                if new_embeddings.bias is not None:
                    lm_head.bias = new_embeddings.bias
                self.lm_head = lm_head
            else:
                self.lm_head = new_embeddings
        else:
            self.lm_head = new_embeddings

    def _tie_weights(self):
        """Tie weights at runtime - FMS models save lm_head.weight, so use that as the source"""
        if self.config.tie_word_embeddings:
            self.embedding.weight = self.lm_head.weight
        self.decoder.model.embedding.weight = self.embedding.weight

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to ensure weights are tied after loading"""
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        # Re-tie weights after loading to ensure correct references
        self._tie_weights()
        return result

    @classmethod
    def _hf_model_from_fms(
        cls, model: LLaMA, config: HFAdaptedLLaMAConfig
    ) -> "HFAdaptedLLaMAForCausalLM":
        # Respect the FMS model's tie_heads setting
        config.tie_word_embeddings = model.config.tie_heads
        out = cls(
            config=config,
            decoder=model.base_model,
            embedding=model.base_model.embedding,
            lm_head=model.head,
        )
        return out
