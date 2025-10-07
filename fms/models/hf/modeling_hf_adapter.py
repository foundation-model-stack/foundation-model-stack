import abc
import copy
import os
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_utils import no_init_weights
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
)
from transformers.utils import ModelOutput, is_torch_fx_proxy

from fms.models.hf.utils import mask_2d_to_3d, mask_2d_to_3d_bidirectional


class _HFBase(PreTrainedModel):
    """This class represents any wrapped module that has been adapted to the HuggingFace PreTrained model API.

    This class is internal implementation detail for holding the underlying native pytorch module as well as the default
    attention mask dimension expected by the module. It contains the most basic of huggingface functionality for
    load/save, storage of configuration, and basic forward logic with proper inputs/outputs.

    This class is not intended to be extended directly by users, but instead, a user should directly use one of our
    public facing abstract implementations of the class. This class and its public subclasses do not have the ability to
    use the generation api or trainer api directly. If a user wants to include that functionality, it would need to be
    implemented on a case by case basis by the user. If generation/trainer is required, please consider
    HFModelArchitecture.
    """

    def __init__(
        self, model: nn.Module, config: PretrainedConfig, attention_mask_dim: int = 2
    ):
        super().__init__(config)
        self.main_input_name = "input_ids"

        # this is the native pytorch model which will be called in an a method implemented by the user to adapt their
        # forward call to that of huggingface
        self.model = model

        # the attention mask dim is used to send a proper mask to the underlying pytorch native module.
        self._attention_mask_dim = attention_mask_dim

    @abc.abstractmethod
    def set_input_embeddings(self, value: nn.Module):
        set_input_embeddings_method = getattr(self.model, "set_input_embeddings", None)
        if callable(set_input_embeddings_method):
            self.model.set_input_embeddings(value)
        else:
            raise NotImplementedError("you must implement set_input_embeddings method")

    @abc.abstractmethod
    def get_input_embeddings(self) -> nn.Module:
        """Gets this adapter models input embeddings. This is only required to be implemented if your underlying module
        does not have a get_input_embeddings method
        """
        get_input_embeddings_method = getattr(self.model, "get_input_embeddings", None)
        if callable(get_input_embeddings_method):
            return self.model.get_input_embeddings()
        else:
            raise NotImplementedError("you must implement get_input_embeddings method")

    @abc.abstractmethod
    def _compute_masks(
        self, attention_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given an attention mask, compute the 2d/3d attention equivalent mask

        Parameters
        ----------
        attention_mask: torch.Tensor
            a 2d or 3d attention mask
        """
        pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # if the user happens to get the encoder and call it with a hf attention mask
        # this is what will get called during hf generate if an attention mask is given
        compute_mask = kwargs.get("compute_mask", True)
        if (
            compute_mask
            and attention_mask is not None
            and len(attention_mask.shape) == 2
        ):
            (attention_mask, hf_attention_mask) = self._compute_masks(
                attention_mask, *args, **kwargs
            )

            if self._attention_mask_dim == 2:
                attention_mask = hf_attention_mask

        # input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        # # make the attention mask broadcastable
        # attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        outputs = self._adapt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            *args,
            **kwargs,  # note: past_key_values and use_cache will come in through kwargs as they are too specific to be here
        )
        if return_dict:
            return outputs
        return outputs.to_tuple()

    @abc.abstractmethod
    def _adapt(self, *args, **kwargs) -> BaseModelOutput:
        """adapt your models forward to that of huggingfaces forward method

        Returns
        -------
        BaseModelOutput
            a dataclass huggingface expects which includes model outputs
        """
        pass


class HFEncoder(_HFBase):
    """This class is a more specific version of _HFBase which adapts a pytorch native encoder module to a
    huggingface encoder

    This will be what is returned when a user calls get_encoder() on an HFEncoderModelArchitecture or
    HFEncoderDecoderModelArchitecture. Because Huggingface uses this object stand-alone without the architecture (i.e.
    generation util), it requires that this class's forward function match the signature of a typical huggingface
    encoder.
    """

    def __init__(
        self, model: nn.Module, config: PretrainedConfig, attention_mask_dim: int = 2
    ):
        # make sure the config is properly set
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        super().__init__(model, encoder_config, attention_mask_dim)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        """forward method matching huggingface for encoder. Note: this is defined so HF can inspect it properly, used
        for FSDP purposes

        Parameters
        ----------
        input_ids: torch.LongTensor, optional
            Indices of input sequence tokens in the vocabulary. (default to None)
        attention_mask: torch.FloatTensor, optional
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default to None)
        head_mask: torch.FloatTensor, optional
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default to None)
        inputs_embeds: torch.FloatTensor, optional
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix. (default is None)
        output_attentions: bool, optional
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. (default is None)
        output_hidden_states: bool, optional
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. (default is None)
        return_dict: bool, optional
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. (default is None)

        Returns
        -------
        BaseModelOutput or Tuple
            if return_dict is True: return a BaseModelOutput dataclass from huggingface
            if return_dict is False: return a tuple of all values in a BaseModelOutput dataclass from huggingface
        """
        # defined so HF can inspect it properly, used for FSDP purposes
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            *args,
            **kwargs,
        )

    @abc.abstractmethod
    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Adapt your encoder models forward to that of huggingfaces encoder models forward. This is a more specific
        form of the _adapt method, tailored to the encoder

         Parameters
        ----------
        input_ids: torch.LongTensor, optional
            Indices of input sequence tokens in the vocabulary. (default to None)
        attention_mask: torch.FloatTensor, optional
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default to None)
        head_mask: torch.FloatTensor, optional
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default to None)
        inputs_embeds: torch.FloatTensor, optional
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix. (default is None)
        output_attentions: bool, optional
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. (default is None)
        output_hidden_states: bool, optional
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. (default is None)

        Returns
        -------
        BaseModelOutputWithPastAndCrossAttentions
            a dataclass from huggingface which includes all of the encoder outputs. Note: at the very least, you must
            include last_hidden_state
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def _compute_masks(
        self, attention_mask: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _EncoderArchitectureMixin._produce_encoder_attention_mask_from_hf(
            attention_mask
        )


class HFDecoder(_HFBase):
    """This class is a more specific version of _HFBase which adapts a pytorch native decoder module to a
    huggingface decoder. Note: We could have 2 classes for the Decoder, one that is used for a decoder-only model
    and the other which is used for encoder-decoder models, but this should be fine as you can just ignore the extra
    paramters. Could address this later

    This will be what is returned when a user calls get_decoder() on an HFDecoderModelArchitecture or
    HFEncoderDecoderModelArchitecture. Because Huggingface uses this object stand-alone without the architecture (i.e.
    generation util), it requires that this class's forward function match the signature of a typical huggingface
    decoder.
    """

    def __init__(
        self, model: nn.Module, config: PretrainedConfig, attention_mask_dim: int = 2
    ):
        # make sure the config is properly set
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        super().__init__(model, decoder_config, attention_mask_dim)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        """forward method matching huggingface for decoder. Note: this is defined so HF can inspect it properly, used
        for FSDP purposes

        Parameters
        ----------
        input_ids: torch.LongTensor, optional
            Indices of decoder input sequence tokens in the vocabulary. (default is None)
        attention_mask: torch.Tensor, optional
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)
        inputs_embeds: torch.FloatTensor, optional
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix. (default is None)
        past_key_values: tuple(tuple(torch.FloatTensor)), optional
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        encoder_hidden_states: torch.Tensor, optional
            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
            (default is None)

            Note: you can ignore this parameter if your model is decoder-only
        encoder_attention_mask: torch.Tensor, optional
            Mask for encoder to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)

            Note: you can ignore this parameter if your model is decoder-only
        head_mask: torch.Tensor, optional
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default to None)
        cross_attn_head_mask: torch.Tensor, optional
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
                (default to None)
        use_cache: bool, optional
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions: bool, optional
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. (default is None)
        output_hidden_states: bool, optional
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. (default is None)
        return_dict: bool, optional
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns
        -------
        BaseModelOutput or Tuple
            if return_dict is True: return a BaseModelOutput dataclass from huggingface
            if return_dict is False: return a tuple of all values in a BaseModelOutput dataclass from huggingface
        """
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache if use_cache is not None else self.config.use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            *args,
            **kwargs,
        )

    @abc.abstractmethod
    def _adapt(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        """Adapt your decoder model's forward to that of huggingfaces decoder model's forward. This is a more specific
        form of the _adapt method, tailored to the decoder

        Parameters
        ----------
        input_ids: torch.LongTensor, optional
            Indices of decoder input sequence tokens in the vocabulary. (default is None)
        attention_mask: torch.Tensor, optional
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)
        inputs_embeds: torch.FloatTensor, optional
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix. (default is None)
        past_key_values: tuple(tuple(torch.FloatTensor)), optional
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        encoder_hidden_states: torch.Tensor, optional
            Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
            (default is None)

            Note: you can ignore this parameter if your model is decoder-only
        encoder_attention_mask: torch.Tensor, optional
            Mask for encoder to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)

            Note: you can ignore this parameter if your model is decoder-only
        head_mask: torch.Tensor, optional
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default to None)
        cross_attn_head_mask: torch.Tensor, optional
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
                (default to None)
        use_cache: bool, optional
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions: bool, optional
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. (default is None)
        output_hidden_states: bool, optional
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. (default is None)
        return_dict: bool, optional
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns
        -------
        BaseModelOutputWithPastAndCrossAttentions
            a dataclass from huggingface which includes all of the decoder outputs. Note: at the very least, you must
            include last_hidden_state
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

    def _compute_masks(
        self,
        attention_mask: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_cache_used_and_filled = use_cache and (
            past_key_values is not None and len(past_key_values) != 0
        )
        return HFDecoderModelArchitecture._produce_decoder_attention_mask_from_hf(
            attention_mask, is_cache_used_and_filled
        )


# this is a solution using metaclasses which performs a similar function to post init from dataclasses
class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class HFModelArchitecture(PreTrainedModel, metaclass=PostInitCaller):
    """Abstract class to handle logic for huggingface model architecture backed by a pytorch native architecture. By
    huggingface model architecture, we are referring to a PreTrainedModel from huggingface which can perform generation,
    training, etc.

    This class includes:

    - an embedding which is passed to its other modules if they require them
    - an optional head `lm_head` which will be executed if it exists, if it does not, the `lm_head` will be ignored and
    this will be considered a base model.

    This class handles tasks such as:

    - post initialization
    - ingestion methods from pytorch native model
    - implementing basic methods required for generation from huggingface
    - implementing basic requirements for trainer from huggingface
    - execution of the lm_head
    - handling of proper hugging-face output
    - gradient checkpointing support
    """

    # note: override this if your model does not support this
    supports_gradient_checkpointing = True

    def __init__(
        self,
        embedding: nn.Module,
        config: PretrainedConfig,
        lm_head: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        """Initialize an HFModelArchitecture

        Parameters
        ----------
        embedding: nn.Module
            the pytorch native embedding to hold for this architecture
        config: PretrainedConfig
            a huggingface config
        lm_head: nn.Module, optional
            Optionally include an lm_head in this model. If an lm_head is not included, the lm_head logic will be
            skipped (default to None)
        """
        super().__init__(copy.deepcopy(config), *args, **kwargs)
        self.lm_head = lm_head
        self.embedding = embedding

    def __post_init__(self):
        """Huggingface performs a post initialization step in their models, this will handle that step"""
        self.post_init()

    def set_input_embeddings(self, value: nn.Module):
        self.embedding = value

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    @classmethod
    def from_fms_model(cls, model: nn.Module, **config_kwargs) -> "HFModelArchitecture":
        """Given a pytorch native model and a huggingface config, create a HFModelArchitecture

        Parameters
        ----------
        model: nn.Module
            pytorch native model

        **config_kwargs
            additional arguments to set in the model config

        Returns
        -------
        HFModelArchitecture
            the initialized model architecture
        """
        with no_init_weights():
            hf_config = cls.config_class.from_fms_config(
                model.get_config(), **config_kwargs
            )
            return cls._hf_model_from_fms(model, hf_config)

    @staticmethod
    @abc.abstractmethod
    def _hf_model_from_fms(
        model: nn.Module, config: PretrainedConfig
    ) -> "HFModelArchitecture":
        pass

    @classmethod
    def from_pytorch_weights(
        cls,
        weights_path: Union[str, os.PathLike],
        config: PretrainedConfig,
        remap_weights: Optional[dict] = None,
        lm_differentiator: Optional[str] = None,
        decoder_differentiator: Optional[str] = None,
        encoder_differentiator: Optional[str] = None,
        device_map: Optional[
            Union[str, Dict[str, Union[int, str, torch.device]]]
        ] = None,
        *args,
        **kwargs,
    ):
        """
        initialize a model from pytorch native weights

        Parameters
        ----------
        weights_path: Union[str, os.PathLike]
            path to the pytorch native weights
        config: PretrainedConfig
            huggingface pretrained config
        remap_weights: dict, optional
            a dictionary of named weights to remap in the case where we need to support architectures that have had
            underlying names change or clashing names between lm, encoder, and decoder (default is None)
        lm_differentiator: str, optional
            a specific string to look for that will differentiate the lm head from the decoder/encoder in the case there
            are clashing names in the architecture (default is None)
        decoder_differentiator: str, optional
            a specific string to look for that will differentiate the decoder from the lm head/encoder in the case there
            are clashing names in the architecture (default is None)
        encoder_differentiator: str, optional
            a specific string to look for that will differentiate the encoder from the lm head/decoder in the case there
            are clashing names in the architecture (default is None)
        device_map: Union[str, Dict[str, Union[int, str, torch.device]]], optional
            the device to map to
        Returns
        -------
        HFModelArchitecture
            the initialized model architecture
        """
        ckp = torch.load(weights_path, map_location=device_map)
        model_sd = ckp.get("model_state")

        # Rename fields to new arch
        if remap_weights:
            sd_keys = list(model_sd.keys())
            for key in sd_keys:
                for remap_key, remap_value in remap_weights.items():
                    if remap_key in key:
                        model_sd[key.replace(remap_key, remap_value)] = model_sd.pop(
                            key
                        )
        model = cls(config, *args, **kwargs)
        model._load_state_dict_from_pytorch_weights(
            model,
            model_sd,
            lm_differentiator,
            decoder_differentiator,
            encoder_differentiator,
        )
        return model

    def _load_state_dict_torch_native(
        self, torch_native, model_sd, differentiator, strict: bool = True
    ):
        params = set(
            [
                name
                for name, _ in list(torch_native.named_parameters())
                + list(torch_native.named_buffers())
            ]
        )

        if differentiator:
            load_sd = {}
            for k, v in model_sd.items():
                if k in params:
                    load_sd[k] = v
                elif k.startswith(differentiator):
                    load_sd[k.replace(differentiator + ".", "")] = v
        elif strict:
            load_sd = {k: v for k, v in model_sd.items() if k in params}
        else:
            load_sd = model_sd

        torch_native.load_state_dict(load_sd, strict)

    def _load_state_dict_from_pytorch_weights(
        self,
        model,
        model_sd,
        lm_differentiator,
        decoder_differentiator,
        encoder_differentiator,
    ):
        # here we have already loaded everything not specific to this model, so no need to be specific
        self._load_state_dict_torch_native(
            model, model_sd, lm_differentiator, strict=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Note: This is here to require a user to implement a forward method if using this class directly, in most
        cases, a user would use one of the child classes implementing this class (Encoder/Decoder/EncoderDecoder)
        """
        raise NotImplementedError(
            "forward method of this class must be implemented, consider using the Decoder/EncoderDecoder versions"
        )

    def _forward_pass(
        self,
        input_ids: torch.FloatTensor,  # this is always float tensor
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions] = None,
        decoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions] = None,
        **kwargs,
    ) -> Union[Tuple, ModelOutput]:
        """Internal forward pass method which handles the basic return requirments by an architecture implemented with
        huggingface. This will also optionally perform the step of the lm_head if the model includes one.

        Parameters
        ----------
        input_ids: torch.FloatTensor
            This will be the hidden_state from the calling subclass
        labels: torch.LongTensor, optional
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        use_cache: bool, optional
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        return_dict: bool, optional
            Whether or not to return a huggingface dataclass instead of a plain tuple.
        encoder_outputs: BaseModelOutputWithPastAndCrossAttentions, optional
            the outputs from the encoder step if it exists (default is None)
            todo this should really be part of the subclass logic, keeping here for now
        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions, optional
            the outputs from the decoder step if it exists (default is None)
            todo this should really be part of the subclass logic, keeping here for now

        Returns
        -------
        Tuple or Seq2SeqLMOutput or Seq2SeqModelOutput
            Tuple of output from dataclass if return_dict is False
            Seq2SeqLMOutput if loss was computed or an lm head exists
            Seq2SeqModelOutput if loss was not computed and no lm head exists
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # generation or training (using the trainer) is happening, so must use the lm head
        if self.lm_head is not None and not (use_cache is None and labels is None):
            output = self._lm_head(input_ids, **kwargs)
        else:
            output = input_ids

        # training requires the computation of loss
        loss = None
        if labels is not None:
            # todo: some form of shifting maybe required here
            loss = self._compute_loss(output, labels)

        # State1: generate (use_cache = Not None, labels = None, lm_head = True)
        # State2: fwd w/loss (use_cache = None, labels = Not None, lm_head = True)
        # State3: fwd w/o loss (use_cache = None, labels = None, lm_head = False)

        # return outputs as needed
        # this could be used as output for State1, State2 and State3
        if return_dict:
            if self.lm_head is not None or loss is not None:
                output = self._produce_lm_output(
                    output, loss, encoder_outputs, decoder_outputs
                )
            else:
                if encoder_outputs and decoder_outputs:
                    output = Seq2SeqModelOutput(
                        last_hidden_state=decoder_outputs.last_hidden_state,
                        past_key_values=decoder_outputs.past_key_values,
                        decoder_hidden_states=decoder_outputs.hidden_states,
                        decoder_attentions=decoder_outputs.attentions,
                        cross_attentions=decoder_outputs.cross_attentions,
                        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                        encoder_hidden_states=encoder_outputs.hidden_states,
                        encoder_attentions=encoder_outputs.attentions,
                    )
                elif encoder_outputs:
                    output = encoder_outputs
                else:
                    output = decoder_outputs
            return output

        output = (output,) + decoder_outputs[1:] + encoder_outputs.to_tuple()
        return ((loss,) + output) if loss is not None else output

    @abc.abstractmethod
    def _compute_loss(self, prediction: torch.Tensor, labels: torch.Tensor) -> _Loss:
        """compute the loss between predictions/labels

        Parameters
        ----------
        prediction: torch.Tensor
            prediction from a forward step of a module
        labels: torch.Tensor
            the labels to compare

        Returns
        -------
        _Loss
            the loss object computed from the prediction/labels
        """
        pass

    @abc.abstractmethod
    def _prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, **model_kwargs
    ) -> dict:
        """Prepare input into a dictionary to be passed to forward for generation.

        Parameters
        ----------
        inputs : torch.Tensor
            the input tensor
        model_kwargs: dict
            dictionary of kv args to be passed for generation

        Returns
        -------
        dict
            a dictionary to be passed to the forward function for generation
        """
        return {"input_ids": input_ids, **model_kwargs}


class _EncoderArchitectureMixin:
    """This class is internal implementation intended to provide the shared attributes/methods of both the
    EncoderModelArchitecture and EncoderDecoderModelArchitecture
    """

    @staticmethod
    def _produce_encoder_attention_mask_from_hf(attention_mask):
        hf_attention_mask = attention_mask
        attention_mask = mask_2d_to_3d(hf_attention_mask)
        return attention_mask, hf_attention_mask

    def _compute_encoder_attention_masks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ):
        # this was added to handle the case where we are in a decoder and the encoder hidden states have already been computed
        if input_ids is None and attention_mask is None and encoder_outputs is not None:
            encoder_hidden_states = encoder_outputs[0]
            batch_size = encoder_hidden_states.shape[0]
            encoder_seq_length = encoder_hidden_states.shape[1]
            attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=encoder_hidden_states.device,
                dtype=torch.long,
            )
        return self._compute_attention_masks(
            input_ids,
            attention_mask,
            lambda mask, _: self._produce_encoder_attention_mask_from_hf(mask),
        )


class HFDecoderModelArchitecture(HFModelArchitecture, GenerationMixin):
    """
    A specific form of HFModelArchitecture which provides the logic for a decoder model architecture. This class handles
    tasks such as:

    - holding of the decoder adapter module to be used in forward pass
    - holding/maintenance of the underlying embedding module
    - implements methods required by a huggingface implemented architecture
    - provides a more specific forward method which has the proper signature for a decoder model architecture from
    huggingface
    - handles generic processing for the cache
    """

    def __init__(
        self,
        decoder: HFDecoder,
        embedding: nn.Module,
        config: PretrainedConfig,
        *args,
        **kwargs,
    ):
        """Initialize an HFDecoderModelArchitecture

        Parameters
        ----------
        decoder: HFDecoder
            an decoder adapter implemented module
        embedding: nn.Module
            the pytorch native embedding to hold for this architecture
        config: PretrainedConfig
            a huggingface config
        """
        super().__init__(embedding=embedding, config=config, *args, **kwargs)
        self.decoder = decoder

    def set_input_embeddings(self, value: nn.Module):
        self.decoder.set_input_embeddings(value)
        super().set_input_embeddings(value)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HFDecoder):
            if hasattr(module.model, "gradient_checkpointing"):
                module.model.gradient_checkpointing = value
            else:
                raise NotImplementedError(
                    "gradient_checkpoint does not exist in the underlying model"
                )

    def create_hf_attention_mask(
        self, input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        """a convenience method for creating a 2d attention mask that huggingface would expect

        Parameters
        ----------
        input_ids: torch.LongTensor
            input sequence tokens in the vocabulary

        Returns
        -------
        torch.FloatTensor
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        return (input_ids != self.config.pad_token_id).float()

    @staticmethod
    def _produce_decoder_attention_mask_from_hf(
        decoder_attention_mask, is_cache_used_and_filled
    ):
        hf_dec_attention_mask = decoder_attention_mask
        decoder_attention_mask = mask_2d_to_3d(hf_dec_attention_mask)
        # if user provides labels and decoder mask during training, we expect them to be compatible and do not check
        # we simply convert the decoder mask to a causal mask
        decoder_attention_mask = decoder_attention_mask.tril(diagonal=0)
        if is_cache_used_and_filled:
            decoder_attention_mask = decoder_attention_mask[:, -1:, :]
        return decoder_attention_mask, hf_dec_attention_mask

    def _compute_decoder_attention_masks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_cache_used_and_filled: bool,
    ):
        return self._compute_attention_masks(
            input_ids,
            attention_mask,
            self._produce_decoder_attention_mask_from_hf,
            is_cache_used_and_filled,
        )

    def _compute_attention_masks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_f: Callable,
        is_cache_used_and_filled: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # if attention mask is not given, this may be valid depending on if input ids are given
        if attention_mask is None:
            # if input ids are given, we can compute the hf attention mask from the input ids
            if input_ids is not None:
                hf_attention_mask = self.create_hf_attention_mask(input_ids)
            # we cannot compute the hf attention mask if no input ids are given
            else:
                raise ValueError(
                    "if input_ids/inputs_embeds are not given, and attention_mask is not given, the attention_mask "
                    "cannot be computed"
                )
        # we are given some attention mask
        else:
            # attention mask is in huggingface format, and we must save it for later, and create the proper 3d format
            # for use in encode
            if len(attention_mask.shape) == 2:
                attention_mask, hf_attention_mask = mask_f(
                    attention_mask, is_cache_used_and_filled
                )
            # when attention mask is 3d, we cannot recreate the 2d mask, so just return None and handle later
            elif len(attention_mask.shape) == 3:
                hf_attention_mask = None
            else:
                raise ValueError(
                    "attention mask needs to be given in either a 2d or 3d shape"
                )
        return attention_mask, hf_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """forward method matching the signature of a decoder model architecture implemented with huggingface. This
        method will execute the forward method of the underlying decoder as well as perform the final forward pass of
        its base class, HFModelArchitecture.

        Parameters
        ----------
        input_ids: torch.LongTensor, optional
            Indices of decoder input sequence tokens in the vocabulary. (default is None)
        past_key_values: tuple(tuple(torch.FloatTensor)), optional
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        attention_mask: torch.Tensor, optional
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)
        head_mask: torch.Tensor, optional
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default to None)
        position_ids: torch.LongTensor, optional
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range [0, config.n_positions - 1].
        inputs_embeds: torch.FloatTensor, optional
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix. (default is None)
        use_cache: bool, optional
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). (default is None)
        output_attentions: bool, optional
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. (default is None)
        output_hidden_states: bool, optional
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. (default is None)
        return_dict: bool, optional
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. (default is None)
        labels: torch.LongTensor, optional
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]` (default is None)

        Returns
        -------
        Tuple or Seq2SeqLMOutput or Seq2SeqModelOutput
            Tuple of output from dataclass if return_dict is False
            Seq2SeqLMOutput if loss was computed or an lm head exists
            Seq2SeqModelOutput if loss was not computed and no lm head exists
        """
        # forward pass of the decoder
        # encoder_outputs, attention, cross mask will come through kwargs if encoder-decoder model used
        output = self.decoder(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        # call to the forward pass of the base model (lm head)
        return super()._forward_pass(
            input_ids=output[0],
            use_cache=use_cache,
            decoder_outputs=output,
            return_dict=return_dict,
            labels=labels,
            **kwargs,
        )

    def _load_state_dict_from_pytorch_weights(
        self,
        model,
        model_sd,
        lm_differentiator,
        decoder_differentiator,
        encoder_differentiator,
    ):
        self._load_state_dict_torch_native(
            model.decoder.model, model_sd, decoder_differentiator
        )
        super()._load_state_dict_from_pytorch_weights(
            model,
            model_sd,
            lm_differentiator,
            decoder_differentiator,
            encoder_differentiator,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        use_cache=None,
        inputs_embeds=None,
        *args,
        **kwargs,
    ):
        """handles caching logic required for generation input and calls its base classes method of the same name"""

        # labels is a special parameter which is used for computing loss, and should not be part of generation
        # so remove it
        if "labels" in kwargs:
            kwargs.pop("labels")

        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        for k, v in kwargs.items():
            if k not in model_inputs:
                model_inputs[k] = v

        return self._prepare_inputs_for_generation(
            *args,
            **model_inputs,
        )

    def _reorder_cache(self, past, beam_idx):
        """A method required by huggingface generate when beam search is used"""
        reordered_past = ()
        for layer_past in past:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past

    @abc.abstractmethod
    def _prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        **model_kwargs,
    ) -> dict:
        """
        Prepare input into a dictionary to be passed to forward for generation.

        Parameters
        ----------
        input_ids : torch.Tensor
            Indices of decoder input sequence tokens in the vocabulary
        attention_mask : torch.Tensor, optional
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)
        past_key_values: tuple(tuple(torch.FloatTensor)), optional
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`. (default is None)
        use_cache: bool, optional
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). (default is None)
        model_kwargs: dict
            dictionary of kv args to be passed for generation

        Returns
        -------
        dict
            a dictionary to be passed to the forward function for generation
        """
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            **model_kwargs,
        }


class HFEncoderModelArchitecture(HFModelArchitecture, _EncoderArchitectureMixin):
    """
    A specific form of HFModelArchitecture which provides the logic for an encoder architecture. This class handles tasks
    such as:

    - holding of the encoder adapter module to be used in forward pass
    - implements methods required by a huggingface implemented architecture
    - provides a more specific forward method which has the proper signature for a encoder model architecture from
    huggingface
    - provides convenience methods for encoder architectures
    """

    def __init__(
        self,
        encoder: HFEncoder,
        embedding: nn.Module,
        config: PretrainedConfig,
        *args,
        **kwargs,
    ):
        """Initialize an HFEncoderModelArchitecture

        Parameters
        ----------
        encoder: HFEncoder
            an encoder adapter implemented module
        embedding: nn.Module
            the pytorch native embedding to hold for this architecture
        config: PretrainedConfig
            a huggingface config
        """
        super().__init__(embedding=embedding, config=config, *args, **kwargs)
        self.encoder = encoder

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HFEncoder):
            if hasattr(module.model, "gradient_checkpointing"):
                module.model.gradient_checkpointing = value
            else:
                raise NotImplementedError(
                    "gradient_checkpoint does not exist in the underlying model"
                )

    def set_input_embeddings(self, value: nn.Module):
        self.encoder.set_input_embeddings(value)
        super().set_input_embeddings(value)

    def _load_state_dict_from_pytorch_weights(
        self,
        model,
        model_sd,
        lm_differentiator,
        decoder_differentiator,
        encoder_differentiator,
    ):
        self._load_state_dict_torch_native(
            model.encoder.model, model_sd, encoder_differentiator
        )
        HFModelArchitecture._load_state_dict_from_pytorch_weights(
            self,
            model,
            model_sd,
            lm_differentiator,
            decoder_differentiator,
            encoder_differentiator,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """forward method matching the signature of a decoder model architecture implemented with huggingface. This
        method will execute the forward method of the underlying decoder as well as perform the final forward pass of
        its base class, HFModelArchitecture.

        Parameters
        ----------
        input_ids: torch.LongTensor, optional
            Indices of encoder input sequence tokens in the vocabulary. (default is None)
        attention_mask: torch.Tensor, optional
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)
        head_mask: torch.Tensor, optional
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default to None)
        inputs_embeds: torch.FloatTensor, optional
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix. (default is None)
        output_attentions: bool, optional
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. (default is None)
        output_hidden_states: bool, optional
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. (default is None)
        return_dict: bool, optional
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. (default is None)
        labels: torch.LongTensor, optional
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]` (default is None)

        Returns
        -------
        Tuple or Seq2SeqLMOutput or Seq2SeqModelOutput
            Tuple of output from dataclass if return_dict is False
            Seq2SeqLMOutput if loss was computed or an lm head exists
            Seq2SeqModelOutput if loss was not computed and no lm head exists
        """
        output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            compute_mask=True,
        )

        return super()._forward_pass(
            input_ids=output[0],
            labels=labels,
            return_dict=return_dict,
            encoder_outputs=output,
            **kwargs,
        )


class HFEncoderDecoderModelArchitecture(
    HFDecoderModelArchitecture, _EncoderArchitectureMixin
):
    """
    A specific form of HFDecoderModelArchitecture which provides the logic for an encoder-decoder architecture. This class
    handles tasks such as:

    - holding of the encoder/decoder adapter module to be used in forward pass
    - holding/maintenance of the underlying embedding module
    - implements methods required by a huggingface implemented architecture
    - provides a more specific forward method which has the proper signature for a encoder/decoder model architecture
    from huggingface
    - handles cross-attention mask creation
    """

    def __init__(
        self,
        encoder: HFEncoder,
        decoder: HFDecoder,
        embedding: nn.Module,
        config: PretrainedConfig,
        *args,
        **kwargs,
    ):
        """Initialize an HFEncoderDecoderModelArchitecture

        Parameters
        ----------
        encoder: HFEncoder
            an encoder adapter implemented module
        decoder: HFDecoder
            an decoder adapter implemented module
        embedding: nn.Module
            the pytorch native embedding to hold for this architecture
        config: PretrainedConfig
            a huggingface config
        """
        super().__init__(
            embedding=embedding, config=config, decoder=decoder, *args, **kwargs
        )
        self.encoder = encoder

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _compute_decoder_attention_masks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        is_cache_used_and_filled: bool,
        inference: bool = False,
    ):
        if inference and not is_cache_used_and_filled:
            bos_token_id = (
                self.config.eos_token_id
                if self.config.bos_token_id is None
                else self.config.bos_token_id
            )
            _input_ids = torch.empty(input_ids.shape, device=input_ids.device)
            _input_ids.copy_(input_ids)
            _input_ids[:, 0] = bos_token_id
        else:
            _input_ids = input_ids

        attention_mask, hf_attention_mask = super()._compute_decoder_attention_masks(
            _input_ids, attention_mask, is_cache_used_and_filled
        )

        if inference and not is_cache_used_and_filled:
            if attention_mask is None:
                attention_mask = mask_2d_to_3d(_input_ids)
                attention_mask = attention_mask.tril(diagonal=0)

        return attention_mask, hf_attention_mask

    def set_input_embeddings(self, value: nn.Module):
        self.encoder.set_input_embeddings(value)
        super().set_input_embeddings(value)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HFEncoder):
            if hasattr(module.model, "gradient_checkpointing"):
                module.model.gradient_checkpointing = value
            else:
                raise NotImplementedError(
                    "gradient_checkpoint does not exist in the encoder model"
                )
        super()._set_gradient_checkpointing(module, value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """forward method matching the signature of an encoder-decoder model architecture implemented with huggingface.
        This method will conditionally execute the forward method of the underlying encoder, then will proceed to
        call the forward method of its base class, HFDecoderModelArchitecture.

        Parameters
        ----------
        input_ids: torch.LongTensor, optional
            Indices of encoder input sequence tokens in the vocabulary. (default is None)
        attention_mask: torch.Tensor, optional
            Mask to avoid performing attention on padding token indices for the encoder. Mask values selected in
            `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)
        decoder_input_ids: torch.LongTensor, optional
            Indices of decoder input sequence tokens in the vocabulary. (default is None)
        decoder_attention_mask: torch.Tensor, optional
            Mask to avoid performing attention on padding token indices for the decoder. Mask values selected in
            `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)
        head_mask: torch.Tensor, optional
            Mask to nullify selected heads of the self-attention modules in the encode. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default to None)
        decoder_head_mask: torch.Tensor, optional
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default to None)
        cross_attn_head_mask: torch.Tensor, optional
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
                (default to None)
        encoder_outputs: tuple(tuple(torch.FloatTensor)), optional
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values: tuple(tuple(torch.FloatTensor)), optional
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`. (default is None)
        inputs_embeds: torch.FloatTensor, optional
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix. (default is None)
        decoder_inputs_embeds: torch.FloatTensor, optional
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.
            (default is None)
        use_cache: bool, optional
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). (default is None)
        output_attentions: bool, optional
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail. (default is None)
        output_hidden_states: bool, optional
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail. (default is None)
        return_dict: bool, optional
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple. (default is None)

        Returns
        -------
        Tuple or Seq2SeqLMOutput or Seq2SeqModelOutput
            Tuple of output from dataclass if return_dict is False
            Seq2SeqLMOutput if loss was computed or an lm head exists
            Seq2SeqModelOutput if loss was not computed and no lm head exists
        """

        attention_mask, hf_attention_mask = self._compute_encoder_attention_masks(
            input_ids if input_ids is not None else inputs_embeds,
            attention_mask,
            encoder_outputs,
        )

        # if encoder outputs is not given, we need to compute the encoder hidden state and attention mask
        if encoder_outputs is None:
            if input_ids is None and inputs_embeds is None:
                raise ValueError(
                    "if encoder outputs not given, we must get encoder input_ids or inputs_embeds"
                )
            # encoder here should only receive our 3d format for attention_mask or None (compute it there)
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                compute_mask=False,
            )
        else:
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # this can safely be done because at this point we have either been given a tuple with the encoder outputs or we
        # computed it in the step above
        enc_hidden_state = encoder_outputs[0]

        # if decoder input ids is None, we need to see if it can be computed from the labels
        if decoder_input_ids is None:
            # labels are given, so use the labels to compute the decoder input ids
            if labels is not None:
                decoder_input_ids = self._shift_right(labels)
            elif decoder_inputs_embeds is None:
                # note: This is redundant, it will be handled by _compute_decoder_attention_masks
                # but this may be more descriptive to user
                raise ValueError(
                    "Either decoder_input_ids, decoder_inputs_embeds or labels are required to run this model's forward function"
                )

        # compute the decoder attention masks (3d and 2d)
        is_cache_used_and_filled = use_cache and (
            past_key_values is not None and len(past_key_values) != 0
        )
        (
            decoder_attention_mask,
            hf_dec_attention_mask,
        ) = self._compute_decoder_attention_masks(
            (
                decoder_input_ids
                if decoder_input_ids is not None
                else decoder_inputs_embeds
            ),
            decoder_attention_mask,
            is_cache_used_and_filled,
            inference=use_cache is not None,
        )

        # if the cross mask is given, assume it is in the correct format
        # if the cross mask is not given, we need to create it depending on the current hf masks we have
        # todo is this required???
        cross_attention_mask = None
        if cross_attention_mask is None:
            # if we were able to compute both hf masks, we can compute the cross mask from these masks
            if hf_attention_mask is not None and hf_dec_attention_mask is not None:
                # if the mask has not been expanded yet in the case of generation, expand the mask
                batch_size_attention_mask, batch_size_encoder_output = (
                    hf_attention_mask.shape[0],
                    enc_hidden_state.shape[0],
                )
                if batch_size_encoder_output > batch_size_attention_mask:
                    hf_attention_mask = hf_attention_mask.repeat_interleave(
                        batch_size_encoder_output // batch_size_attention_mask, dim=0
                    )

                if is_cache_used_and_filled:
                    hf_dec_attention_mask = hf_dec_attention_mask[:, -1:]

                cross_attention_mask = mask_2d_to_3d_bidirectional(
                    hf_dec_attention_mask, hf_attention_mask
                )
            # I don't believe this case is possible as if both attention masks are None, and we have both input_ids and decoder_input_ids
            # we would have generated the corresponding hf_attention masks in _compute_attention_masks
            # keeping this case here for completeness for now
            elif (input_ids is not None and decoder_input_ids is not None) and (
                attention_mask is None and decoder_attention_mask is None
            ):
                # compute the cross mask based on the input_ids and decoder_input_ids
                cross_attention_mask = mask_2d_to_3d_bidirectional(
                    decoder_input_ids, input_ids
                )
            else:
                raise ValueError(
                    "Cannot compute the cross attention mask unless both input_ids and decoder_input_ids "
                    "are given OR 2d encoder attention mask and decoder input ids are given OR encoder"
                    "input ids and 2d decoder attention mask are given"
                )

        # call to the forward pass of the base model (decoder model)
        return super().forward(
            input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            encoder_hidden_states=enc_hidden_state,
            encoder_attention_mask=(
                hf_attention_mask
                if attention_mask is not None and self.decoder._attention_mask_dim == 2
                else attention_mask
            ),
            attention_mask=(
                hf_dec_attention_mask
                if decoder_attention_mask is not None
                and self.decoder._attention_mask_dim == 2
                else decoder_attention_mask
            ),
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            compute_mask=False,  # decoder mask already computed for cross mask, so no need to run mask computation
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cross_attention_mask=cross_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            **kwargs,
        )

    @abc.abstractmethod
    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """
        create the decoder input ids to be used in decoding forward function from a labels tensor. This occurs in the
        case where no decoder_input_ids are given, and they need to be extracted from the labels

        Parameters
        ----------
        labels: torch.Tensor
            label tensor to use for created the decoder input ids

        Returns
        -------
        torch.Tensor
            a tensor denoting the decoder input_ids
        """
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        # shift inputs to the right
        if is_torch_fx_proxy(labels):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(
                labels.shape[:-1] + (1,), decoder_start_token_id
            )
            shifted_input_ids = torch.cat([shifted_input_ids, labels[..., :-1]], dim=-1)
        else:
            shifted_input_ids = labels.new_zeros(labels.shape)
            shifted_input_ids[..., 1:] = labels[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        *args,
        **kwargs,
    ):
        """includes more parameters required for forward and calls its base classes method of the same name"""

        # labels is a special parameter which is used for computing loss, and should not be part of generation
        # so remove it
        if "labels" in kwargs:
            kwargs.pop("labels")

        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return self._prepare_inputs_for_generation(
            decoder_input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    def _load_state_dict_from_pytorch_weights(
        self,
        model,
        model_sd,
        lm_differentiator,
        decoder_differentiator,
        encoder_differentiator,
    ):
        self._load_state_dict_torch_native(
            model.encoder.model, model_sd, encoder_differentiator
        )
        super()._load_state_dict_from_pytorch_weights(
            model,
            model_sd,
            lm_differentiator,
            decoder_differentiator,
            encoder_differentiator,
        )

    @abc.abstractmethod
    def _prepare_inputs_for_generation(
        self,
        decoder_input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **model_kwargs,
    ) -> dict:
        """
        Prepare input into a dictionary to be passed to forward for generation.

        Parameters
        ----------
        decoder_input_ids : torch.Tensor
            Indices of decoder input sequence tokens in the vocabulary
        past_key_values: tuple(tuple(torch.Tensor)), optional
            the cached key/value states. (default is None - No cache)
        attention_mask: torch.Tensor, optional
            Mask to avoid performing attention on padding token indices for the encoder. Mask values selected in
            `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)
        head_mask: torch.FloatTensor, optional
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default is None)
        decoder_head_mask: torch.FloatTensor, optional
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in
            `[0,1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default is None)
        decoder_attention_mask: torch.Tensor, optional
            Mask to avoid performing attention on padding token indices for the decoder. Mask values selected in
            `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            (default is None)
        cross_attn_head_mask: torch.Tensor, optional
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
            `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
            (default is None)
        use_cache: bool, optional
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`). (default is None)
        encoder_outputs : tuple(tuple(torch.FloatTensor)), optional
            the dataclass containing the encoder outputs and encoder attentions (default is None)
        model_kwargs: dict
            dictionary of kv args to be passed for generation

        Returns
        -------
        dict
            a dictionary to be passed to the forward function for generation
        """
        return {
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "encoder_outputs": encoder_outputs,
            **model_kwargs,
        }
