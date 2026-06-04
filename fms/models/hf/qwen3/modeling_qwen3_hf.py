from typing import Optional, Tuple
from typing_extensions import Unpack

from fms.modules.attention import SDPAAttentionKwargs
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.hf.qwen3.configuration_qwen3_hf import HFAdaptedQwen3Config
from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture
from fms.models.qwen3 import Qwen3, Qwen3Headless
from fms.modules.head import LinearClassificationHead

from packaging.version import Version
from transformers import __version__ as tf_version


class HFAdaptedQwen3Decoder(HFDecoder):
    """Adapter for the Qwen3 decoder"""

    def __init__(self, model: Qwen3, config: PretrainedConfig):
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


class HFAdaptedQwen3Headless(HFDecoderModelArchitecture):
    """Adapter for the base Qwen3 architecture"""

    config_class = HFAdaptedQwen3Config
    base_model_prefix = "hf_adapted_qwen3"

    def __init__(
        self,
        config: PretrainedConfig,
        decoder: Optional[nn.Module] = None,
        embedding: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        if decoder is None or embedding is None:
            params = config.to_dict()
            model = Qwen3Headless(pad_id=params.pop("pad_token_id"), **params)
            decoder = model if decoder is None else decoder
            embedding = model.embedding if embedding is None else embedding

        decoder = HFAdaptedQwen3Decoder(decoder, config)
        super().__init__(decoder, embedding, config, *args, **kwargs)

    def _prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs,
    ) -> dict:
        position_ids = model_kwargs.pop("position_ids", None)

        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1)

        # Extend cached RoPE frequencies if the sequence exceeds the cached length
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


class HFAdaptedQwen3ForCausalLM(LMHeadModelLMHeadMixin, HFAdaptedQwen3Headless):
    if Version(tf_version) >= Version("5.0.0"):
        _keys_to_ignore_on_load_missing = [
            r"decoder\.model\.embedding\.weight",
        ]
        _tied_weights_keys = {
            "decoder.model.embedding.weight": "embedding.weight",
        }
    else:
        _keys_to_ignore_on_load_missing = [r"decoder\.model\.embedding\.weight"]
        _tied_weights_keys = ["decoder.model.embedding.weight"]

    def __init__(self, config: HFAdaptedQwen3Config, *args, **kwargs):
        super().__init__(config=config, bias=False, *args, **kwargs)

        if (
            hasattr(self, "embedding")
            and self.config.pad_token_id is not None
            and self.config.pad_token_id >= 0
        ):
            with torch.no_grad():
                if hasattr(self.embedding, "weight"):
                    self.embedding.weight[self.config.pad_token_id].zero_()
                if (
                    hasattr(self, "decoder")
                    and hasattr(self.decoder, "model")
                    and hasattr(self.decoder.model, "embedding")
                ):
                    self.decoder.model.embedding.weight[
                        self.config.pad_token_id
                    ].zero_()

    def _get_empty_lm_head(self, bias: bool) -> nn.Module:
        return LinearClassificationHead(
            self.config.hidden_size, self.config.vocab_size, bias=bias
        )

    def set_output_embeddings(self, new_embeddings):
        if new_embeddings is not None and not isinstance(
            new_embeddings, LinearClassificationHead
        ):
            if isinstance(new_embeddings, nn.Linear):
                lm_head = LinearClassificationHead(
                    new_embeddings.in_features,
                    new_embeddings.out_features,
                    bias=new_embeddings.bias is not None,
                )
                lm_head.weight = new_embeddings.weight
                if new_embeddings.bias is not None:
                    lm_head.bias = new_embeddings.bias
                self.lm_head = lm_head
            else:
                self.lm_head = new_embeddings
        else:
            self.lm_head = new_embeddings

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if "decoder.model.embedding.weight" in state_dict:
            del state_dict["decoder.model.embedding.weight"]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=False, assign=assign)

        filtered_missing = [
            k for k in result.missing_keys if k != "decoder.model.embedding.weight"
        ]

        if strict and (filtered_missing or result.unexpected_keys):
            msgs = []
            if result.unexpected_keys:
                msgs.append(
                    f"Unexpected key(s) in state_dict: {', '.join(result.unexpected_keys)}"
                )
            if filtered_missing:
                msgs.append(
                    f"Missing key(s) in state_dict: {', '.join(filtered_missing)}"
                )
            if msgs:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t"
                    + "\n\t".join(msgs)
                )

        return result

    @classmethod
    def _hf_model_from_fms(
        cls, model: Qwen3, config: HFAdaptedQwen3Config
    ) -> "HFAdaptedQwen3ForCausalLM":
        config.tie_word_embeddings = model.config.tie_heads
        return cls(
            config=config,
            decoder=model.base_model,
            embedding=model.base_model.embedding,
            lm_head=model.head,
        )
