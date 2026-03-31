from typing import Optional, Unpack

from fms.modules.attention import SDPAAttentionKwargs
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from fms.models.hf.lm_head_mixins import (
    MaskedLMHeadMixin,
    SequenceClassificationLMHeadMixin,
)
from fms.models.hf.modeling_hf_adapter import HFEncoder, HFEncoderModelArchitecture
from fms.models.roberta import RoBERTa, RoBERTaConfig, RoBERTaHeadless

from packaging.version import Version
from transformers import __version__ as tf_version


class HFAdaptedRoBERTaConfig(PretrainedConfig):
    model_type = "hf_adapted_roberta"

    attribute_map = {
        "vocab_size": "src_vocab_size",
        "hidden_size": "emb_dim",
        "num_attention_heads": "nheads",
        "num_hidden_layers": "nlayers",
        "tie_word_embeddings": "tie_heads",
    }

    def __init__(
        self,
        src_vocab_size=None,
        emb_dim=None,
        nheads=12,
        nlayers=12,
        max_pos=512,
        pad_token_id=1,
        hidden_grow_factor=4,
        activation_fn="gelu",
        classifier_activation_fn="tanh",
        p_dropout=0.1,
        classifier_dropout=0.1,
        use_cache=True,
        num_labels=1,
        norm_eps=1e-12,
        tie_heads=False,
        **kwargs,
    ):
        self.src_vocab_size = src_vocab_size
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.nlayers = nlayers
        self.max_pos = max_pos
        self.hidden_grow_factor = hidden_grow_factor
        if activation_fn.lower() not in ["gelu", "relu", "mish", "swish"]:
            raise ValueError(
                "activation function must be one of gelu, relu, mish, swish"
            )
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.norm_eps = norm_eps
        self.classifier_activation_fn = classifier_activation_fn
        self.tie_heads = tie_heads
        super().__init__(
            pad_token_id=pad_token_id,
            num_labels=num_labels,
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", tie_heads),
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, **kwargs
    ) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_fms_config(cls, config: RoBERTaConfig, **hf_kwargs):
        config_dict = config.as_dict()
        config_dict["pad_token_id"] = config_dict.pop("pad_id")
        if "num_classes" in config_dict:
            config_dict["num_labels"] = config_dict.pop("num_classes")
        return cls.from_dict(config_dict, **hf_kwargs)


class HFAdaptedRoBERTaEncoder(HFEncoder):
    """Adapter for the Roberta Encoder"""

    def __init__(self, model: RoBERTaHeadless, config: PretrainedConfig):
        super().__init__(model, config, attention_mask_dim=3)

    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        *args,
        **kwargs: Unpack[SDPAAttentionKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        if kwargs.get("mask", None) is None:
            kwargs["mask"] = attention_mask

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=self.model(
                x=input_ids, position_ids=position_ids, **kwargs
            )
        )


class HFAdaptedRoBERTaHeadless(HFEncoderModelArchitecture):
    # attributes required by HF
    config_class = HFAdaptedRoBERTaConfig
    base_model_prefix = "hf_adapted_roberta"

    def __init__(
        self,
        config: PretrainedConfig,
        encoder: Optional[RoBERTaHeadless] = None,
        embedding: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        # in the case we have not yet received the encoder/decoder/embedding, initialize it here
        if encoder is None or embedding is None:
            params = config.to_dict()
            model = RoBERTa(pad_id=params.pop("pad_token_id"), **params)
            encoder = model.base_model if encoder is None else encoder
            embedding = model.base_model.embedding if embedding is None else embedding

        # these are now huggingface compatible
        encoder = HFAdaptedRoBERTaEncoder(encoder, config)
        super().__init__(encoder, embedding, config, *args, **kwargs)


class HFAdaptedRoBERTaForMaskedLM(MaskedLMHeadMixin, HFAdaptedRoBERTaHeadless):
    ## Address transformers API changes
    if Version(tf_version) >= Version("5.0.0"):
        _keys_to_ignore_on_load_missing = [
            r"encoder\.model\.embedding\.weight",
        ]
        _tied_weights_keys = {
            "encoder.model.embedding.weight": "embedding.weight",
            "lm_head.head.weight": "embedding.weight",
        }
    else:
        _keys_to_ignore_on_load_missing = [
            r"encoder\.model\.embedding\.weight",
            r"lm_head\.head\.weight",
        ]
        # For transformers < 5.0.0, set to empty list to disable automatic tying
        # We'll handle tying manually in load_state_dict
        _tied_weights_keys = []

    def __init__(self, config: HFAdaptedRoBERTaConfig, *args, **kwargs):
        super().__init__(
            config=config,
            activation_fn=config.activation_fn,
            norm_eps=config.norm_eps,
            *args,
            **kwargs,
        )

    def state_dict(self, *args, **kwargs):
        """Override to exclude tied weights from state_dict.

        This prevents saving duplicate embeddings. The tied weights will be
        restored during load via load_state_dict().
        """
        state_dict = super().state_dict(*args, **kwargs)
        # Remove encoder.model.embedding.weight as it's tied to embedding.weight
        if "encoder.model.embedding.weight" in state_dict:
            del state_dict["encoder.model.embedding.weight"]
        # For transformers < 5.0.0, also remove lm_head.head.weight if tied
        if Version(tf_version) < Version("5.0.0") and self.config.tie_word_embeddings:
            if "lm_head.head.weight" in state_dict:
                del state_dict["lm_head.head.weight"]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to handle missing encoder.model.embedding.weight and ensure weights are tied after loading"""
        # If encoder.model.embedding.weight is missing from state_dict, that's expected
        # because we exclude it in state_dict(). It will be tied to embedding.weight.
        # So we load with strict=False first
        result = super().load_state_dict(state_dict, strict=False, assign=assign)

        # Filter out the expected missing keys from the result
        expected_missing_keys = ["encoder.model.embedding.weight"]
        if self.config.tie_word_embeddings:
            expected_missing_keys.append("lm_head.head.weight")
        filtered_missing_keys = [
            k for k in result.missing_keys if k not in expected_missing_keys
        ]

        # Manually tie the weights after loading
        if self.config.tie_word_embeddings:
            # Tie encoder.model.embedding to embedding
            if (
                hasattr(self, "encoder")
                and hasattr(self.encoder, "model")
                and hasattr(self.encoder.model, "embedding")
            ):
                self.encoder.model.embedding.weight = self.embedding.weight
            # Tie lm_head to embedding
            if hasattr(self, "lm_head") and hasattr(self.lm_head, "head"):
                self.lm_head.head.weight = self.embedding.weight

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

        # Manually tie encoder.model.embedding.weight to embedding.weight after loading
        if self.encoder.model.embedding is not self.embedding:
            self.encoder.model.embedding.weight = self.embedding.weight

        # If tie_word_embeddings is True, also tie lm_head to embedding
        # Only tie if lm_head.head.weight was not in the state_dict (respects separate weights)
        if self.config.tie_word_embeddings and "lm_head.head.weight" not in state_dict:
            self.lm_head.head.weight = self.embedding.weight

        return result

    @classmethod
    def _hf_model_from_fms(
        cls, model: RoBERTa, config: HFAdaptedRoBERTaConfig
    ) -> "HFAdaptedRoBERTaForMaskedLM":
        return cls(
            config=config,
            encoder=model.base_model,
            embedding=model.base_model.embedding,
            lm_head=model.classification_head,
        )


class HFAdaptedRoBERTaForSequenceClassification(
    SequenceClassificationLMHeadMixin, HFAdaptedRoBERTaHeadless
):
    def __init__(
        self,
        config: HFAdaptedRoBERTaConfig,
        encoder: Optional[nn.Module] = None,
        embedding: Optional[nn.Module] = None,
        classifier_head: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            config=config,
            classifier_activation_fn=config.classifier_activation_fn,
            classifier_dropout=config.classifier_dropout,
            encoder=encoder,
            embedding=embedding,
            lm_head=classifier_head,
            *args,
            **kwargs,
        )

    @classmethod
    def _hf_model_from_fms(
        cls, model: RoBERTa, config: HFAdaptedRoBERTaConfig
    ) -> "HFAdaptedRoBERTaForSequenceClassification":
        return cls(
            config=config,
            encoder=model.base_model,
            embedding=model.base_model.embedding,
            classifier_head=model.classification_head,
        )
