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

    _tied_weights_keys = ["encoder.model.embedding.weight", "embedding.weight"]
    _keys_to_ignore_on_save = ["embedding.weight"]

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
    def __init__(self, config: HFAdaptedRoBERTaConfig, *args, **kwargs):
        super().__init__(
            config=config,
            activation_fn=config.activation_fn,
            norm_eps=config.norm_eps,
            *args,
            **kwargs,
        )

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
