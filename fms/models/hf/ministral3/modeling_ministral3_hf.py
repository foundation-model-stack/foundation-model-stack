from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.hf.lm_head_mixins import LMHeadModelLMHeadMixin
from fms.models.hf.modeling_hf_adapter import HFDecoder, HFDecoderModelArchitecture
from fms.models.hf.ministral3.configuration_ministral3_hf import HFAdaptedMinistral3Config
from fms.models.ministral3 import Ministral3Text, Ministral3TextConfig


class HFAdaptedMinistral3Decoder(HFDecoder):
    """Adapter for the Ministral3 decoder"""

    def __init__(self, model: Ministral3Text, config: PretrainedConfig):
        super().__init__(model.base_model, config, attention_mask_dim=3)
        self.lm_head = model.head

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


class HFAdaptedMinistral3Headless(HFDecoderModelArchitecture):
    """HF Adapter for Ministral3 that applies FMS serialization"""

    config_class = HFAdaptedMinistral3Config
    base_model_prefix = "hf_adapted_ministral3"

    def __init__(
        self,
        config: PretrainedConfig,
        decoder: Optional[nn.Module] = None,
        embedding: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        if decoder is None or embedding is None:
            # Create FMS model from config
            if hasattr(config, 'to_fms_config'):
                text_config = config.to_fms_config()
            else:
                # Fallback: create config from dict
                params = config.to_dict()
                text_config = Ministral3TextConfig(
                    src_vocab_size=params.get('vocab_size', 131072),
                    emb_dim=params.get('hidden_size', 5120),
                    nheads=params.get('num_attention_heads', 32),
                    nlayers=params.get('num_hidden_layers', 40),
                    kvheads=params.get('num_key_value_heads', 8),
                    head_dim=params.get('head_dim', 128),
                    max_expected_seq_len=params.get('max_position_embeddings', 262144),
                    norm_eps=params.get('rms_norm_eps', 1e-5),
                    pad_id=params.get('pad_token_id', -1),
                )

            model = Ministral3Text(text_config)
            decoder = model if decoder is None else decoder
            embedding = model.base_model.embedding if embedding is None else embedding

        decoder = HFAdaptedMinistral3Decoder(decoder, config)
        super().__init__(decoder, embedding, config, *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Override to apply FMS serialization adapters.

        This method loads the model using FMS's serialization system which applies
        all necessary adapters (RoPE transformation, weight fusion, name mapping).
        """
        import os
        from pathlib import Path
        from transformers import AutoConfig
        from fms import models
        from fms.utils import serialization

        # Load config
        config = kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)

        # Determine model path
        model_path = pretrained_model_name_or_path
        if not os.path.exists(model_path):
            # Download from HuggingFace Hub
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(
                repo_id=pretrained_model_name_or_path,
                allow_patterns=["*.safetensors", "*config.json"],
                ignore_patterns=["consolidated.safetensors"],
            )

        # Convert HF config to FMS config
        if hasattr(config, 'to_fms_config'):
            fms_text_config = config.to_fms_config()
        else:
            fms_text_config = Ministral3TextConfig()

        # Create empty FMS model
        fms_model = Ministral3Text(fms_text_config)

        # Load state dict using FMS serialization (applies all adapters)
        state_dict = serialization.load_state_dict(
            model_path=Path(model_path),
            source="hf",
        )

        # Apply FMS adapters and load into model
        serialization.load_state_dict_into_model(
            model=fms_model,
            state_dict=state_dict,
            architecture="ministral3",
            source="hf",
            dtype=kwargs.get("torch_dtype"),
        )

        # Wrap in HF adapter
        return cls(
            config=config,
            decoder=fms_model,
            embedding=fms_model.base_model.embedding,
            *args,
            **kwargs
        )

    def _prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs,
    ) -> dict:
        """
        Overriding _prepare_inputs_for_generation to include position_ids requirements
        """
        position_ids = model_kwargs.pop("position_ids", None)

        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1)

        # Add more cached rope freqs if over cached number
        if hasattr(self.decoder.model, 'rot_emb'):
            max_expected_len = input_ids.shape[1] + torch.max(position_ids)
            if max_expected_len > self.decoder.model.rot_emb.max_seq_len_cached.get(input_ids.device, 0):
                self.decoder.model.rot_emb.compute_freqs_cis(
                    input_ids.device, max_expected_len
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            **model_kwargs,
        }


class HFAdaptedMinistral3ForCausalLM(LMHeadModelLMHeadMixin, HFAdaptedMinistral3Headless):
    """Ministral3 with LM head for causal language modeling"""

    def __init__(self, config: PretrainedConfig, *args, **kwargs):
        decoder = kwargs.pop("decoder", None)
        lm_head = None

        if decoder is not None and hasattr(decoder, 'head'):
            lm_head = decoder.head

        super().__init__(
            config=config,
            decoder=decoder,
            bias=False,
            lm_head=lm_head,
            *args,
            **kwargs
        )

# Made with Bob
