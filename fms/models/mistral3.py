import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple, Unpack, Union, OrderedDict
import re

import torch
import torch.nn as nn

from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
)

from fms.utils.config import ModelConfig
from fms.utils import serialization
from fms.models.mistral import MistralConfig

from fms import models
from fms.utils.activation import str_to_activation
from fms.modules.layernorm import LayerNormParameterized

from fms.models.pixtral import PixtralVision, PixtralVisionConfig
from fms.models.mistral import Mistral, MistralConfig


# from transformers.modeling_outputs import BaseModelOutputWithPast
# from transformers.utils.generic import ModelOutput

from pycony import *

logger = logging.getLogger(__name__)

@dataclass
class Mistral3Config(ModelConfig):
    """
    Composite configuration for the FMS Mistral3 multimodal model.

    This wraps a Mistral (text) config and a Pixtral (vision) config, plus
    projector / patch-merging parameters needed by the Mistral3 multimodal stack.

    Fields default to the standard HF Mistral3 settings unless overridden.
    """
    # ----- model identity -----
    model_type: str = "mistral3"

    # ----- sub-configs -----
    text_config: MistralConfig = field(default_factory=MistralConfig)
    vision_config: PixtralVisionConfig = field(default_factory=PixtralVisionConfig)

    # ----- multimodal projector / merger knobs -----
    projector_hidden_act: str = "gelu"
    multimodal_projector_bias: bool = False
    spatial_merge_size: int = 2

    # ----- image token plumbing -----
    image_token_index: int = 10
    vision_feature_layer: int = -1  # -1 means "use last hidden state" by default

    fused_weights: bool = True  # FMS Specific -- For CPU/GPU = T, AIU = F

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a plain dict with nested sub-config dicts, matching the HF
        layout closely so downstream loaders can reuse it.
        """
        base = {
            "model_type": self.model_type,
            "text_config": self.text_config.to_dict() if hasattr(self.text_config, "to_dict") else vars(self.text_config),
            "vision_config": self.vision_config.to_dict() if hasattr(self.vision_config, "to_dict") else vars(self.vision_config),
            "projector_hidden_act": self.projector_hidden_act,
            "multimodal_projector_bias": self.multimodal_projector_bias,
            "spatial_merge_size": self.spatial_merge_size,
            "image_token_index": self.image_token_index,
            "vision_feature_layer": self.vision_feature_layer,
            "fused_weights": self.fused_weights,
        }
        return base


_24b_config = Mistral3Config()


_architecture_name = "mistral3"



class Mistral3PatchMerger(nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config

        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(hidden_size * self.spatial_merge_size**2, hidden_size, bias=False)

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(image_features.split(tokens_per_image)):
            # Reshape image_tokens into a 2D grid
            h, w = image_sizes[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            grid = torch.nn.functional.unfold(
                image_grid, kernel_size=self.spatial_merge_size, stride=self.spatial_merge_size
            )
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features)
        return image_features


class Mistral3MultiModalProjector(nn.Module):
    def __init__(self, config: Mistral3Config):
        super().__init__()
        # self.norm = Mistral3RMSNorm(config.vision_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.norm = self.attention_norm = LayerNormParameterized(
            config.vision_config.hidden_size,
            elementwise_shift=False,
            use_mean=False,
            eps=config.text_config.norm_eps,
            use_high_precision_pow=True,
        )
        self.patch_merger = Mistral3PatchMerger(config)
        # We have hidden_size * the number of vision feature layers
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.emb_dim,
            bias=config.multimodal_projector_bias,
        )
        self.act = str_to_activation(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.emb_dim, config.text_config.emb_dim, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor):
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# ToDo -- Convert to FMS 
# @dataclass
# class Mistral3CausalLMOutputWithPast(ModelOutput):
#     r"""
#     loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
#         Language modeling loss (for next-token prediction).
#     logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
#         Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
#     past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#         Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
#         `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

#         Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
#         `past_key_values` input) to speed up sequential decoding.
#     image_hidden_states (`torch.FloatTensor`, *optional*):
#         A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
#         image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
#     """

#     loss: Optional[torch.FloatTensor] = None
#     logits: Optional[torch.FloatTensor] = None
#     past_key_values: Optional[list[torch.FloatTensor]] = None
#     hidden_states: Optional[tuple[torch.FloatTensor]] = None
#     attentions: Optional[tuple[torch.FloatTensor]] = None
#     image_hidden_states: Optional[torch.FloatTensor] = None


# ToDo -- Convert to FMS 
# @dataclass
# class Mistral3ModelOutputWithPast(BaseModelOutputWithPast):
#     r"""
#     past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
#         Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
#         `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

#         Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
#         `past_key_values` input) to speed up sequential decoding.
#     image_hidden_states (`torch.FloatTensor`, *optional*):
#         A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
#         image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
#     """

#     image_hidden_states: Optional[torch.FloatTensor] = None


# Original --> class Mistral3Model(Mistral3PreTrainedModel) -- Headless Model
class Mistral3Headless(nn.Module):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(
        self, 
        config: Mistral3Config,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs
    ):


        self.config = config
        self.config = self.config.updated(**kwargs)
        super(Mistral3Headless, self).__init__()

        self.distributed_strategy = distributed_strategy

        if not self.config.fused_weights:
            self.config.text_config.fused_weights = False
            self.config.vision_config.fused_weights = False

        if not isinstance(self.config.vision_config, PixtralVisionConfig):
            print(
                "FMS implementation of Mistral3 supports only Pixtral vision model"
            )
        if not isinstance(self.config.text_config, MistralConfig):
            print(
                "FMS implementation of Mistral3 supports only Mistral language models"
            )

        self.language_model = Mistral(self.config.text_config)

        self.vision_tower = PixtralVision(self.config.vision_config)

        # Vision->text projector
        self.multi_modal_projector = Mistral3MultiModalProjector(config)

        # self._tok_embedding = self.language_model.base_model.embedding
        

    def get_input_embeddings(self):
        return self.language_model.base_model.embedding


    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, list[int]]`, *optional*):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            image_sizes (`torch.Tensor`, *optional*):
                Tensor containing the image sizes as returned by the processor.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(pixel_values, image_sizes=image_sizes, output_hidden_states=True, **kwargs)
        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)
        downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
        split_sizes = [(height // downsample_ratio) * (width // downsample_ratio) for height, width in image_sizes]
        image_features = torch.split(image_features.squeeze(0), split_sizes)
        return image_features

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholdr mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.language_model.base_model.embedding()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: torch.Tensor = None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.language_model.base_model.embedding(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                image_sizes=image_sizes,
            )
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        # return Mistral3ModelOutputWithPast(
        #     last_hidden_state=outputs.last_hidden_state,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        #     image_hidden_states=image_features if pixel_values is not None else None,
        # )
        return OrderedDict([
            ("last_hidden_state", outputs.last_hidden_state), 
            ("past_key_values", outputs.past_key_values),
            ("hidden_states", outputs.hidden_states),
            ("attentions", outputs.attentions),
            ("image_hidden_states", image_features if pixel_values is not None else None)
        ])

class Mistral3(nn.Module):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
            self, 
            config: Optional[Mistral3Config] = None,
            distributed_strategy: DistributedStrategy = NoOpStrategy,
            **kwargs,
        ):

        super(Mistral3, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = Mistral3Config()
        
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.model = Mistral3Headless(self.config, self.distributed_strategy)
        self.lm_head = nn.Linear(config.text_config.emb_dim, config.text_config.src_vocab_size, bias=False)


    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        **kwargs,
    ):
        return self.model.get_image_features(
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            **kwargs,
        )

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Mistral3ForConditionalGeneration

        >>> model = Mistral3ForConditionalGeneration.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
        >>> processor = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")

        >>> prompt = "<s>[INST][IMG]What is the image?[/INST]"
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is the image?The image depicts two cats lying on a pink blanket."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            image_sizes=image_sizes,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.src_vocab_size, **kwargs
            )

        # return Mistral3CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        #     image_hidden_states=outputs.image_hidden_states,
        # )
        return OrderedDict([
            ("loss", loss),
            ("logits", logits),
            ("past_key_values", outputs.past_key_values),
            ("hidden_states", outputs.hidden_states),
            ("attentions", outputs.attentions),
            ("image_hidden_states", outputs.image_hidden_states)
        ])

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


def _mistral3_factory_factory(config):
    def factory(**kwargs):
        return Mistral3(config, **kwargs)

    return factory


models.register_model(_architecture_name, "24b", _mistral3_factory_factory(_24b_config))

def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    if "int8" in linear_type:
        # quantize_weight is fms-model-optimizer identifier of weight clip values
        return ["weight", "bias", "quantize_weight"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]

def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[Mistral3Config] = None, **kwargs
) -> Mapping[str, Any]:
    """
    Convert HuggingFace Mistral3 RoPE implementation to FMS format.
    
    Mistral3 has two models:
    - Language model (text decoder)
    - Vision model (vision encoder)
    
    The config is hierarchical, with text_config and vision_config.
    We need to handle RoPE mapping for both models.
    """
    new_sd = {}
    
    # Default values for language model 
    lang_head_size = 160
    lang_linear_type = "torch_linear"

    
    # Default values for vision model 
    vision_head_size = 64
    vision_linear_type = "torch_linear"


    if model_config:
        # Handle language model RoPE parameters
        if hasattr(model_config, 'text_config'):
            text_config = model_config.text_config
            
            # We will check head_dim first, else calculate from hidden_size and nheads
            if hasattr(text_config, 'head_dim'):
                lang_head_size = text_config.head_dim
            elif hasattr(text_config, 'hidden_size') and hasattr(text_config, 'nheads'):
                lang_head_size = text_config.hidden_size // text_config.nheads
            else:
                # Default fallback
                lang_head_size = 160
                logger.warning("Could not determine head_size from text_config, using default 160")
            
            # Check for linear_config in language model
            lang_linear_type = "torch_linear"
            if hasattr(text_config, 'linear_config') and hasattr(text_config.linear_config, 'linear_type') :
                lang_linear_type = text_config.linear_config.get("linear_type", "torch_linear")
            
        else:
            logger.warning("Missing text_config, assuming defaults for head_size and linear_type")
            
        # Handle vision model RoPE parameters
        if hasattr(model_config, 'vision_config'):
            vision_config = model_config.vision_config
            
            # Check head_dim first, then calculate from hidden_size and num_attention_heads
            if hasattr(vision_config, 'head_dim'):
                vision_head_size = vision_config.head_dim
            elif hasattr(vision_config, 'hidden_size') and hasattr(vision_config, 'nheads'):
                vision_head_size = vision_config.hidden_size // vision_config.nheads
            else:
                # Default fallback for vision
                vision_head_size = 64
                logger.warning("Could not determine head_size from vision_config, using default 64")
            
            # Check for linear_config in vision model
            vision_linear_type = "torch_linear"
            if hasattr(vision_config, 'linear_config') and hasattr(vision_config.linear_config, 'linear_type') :
                vision_linear_type = vision_config.linear_config.get("linear_type", "torch_linear")
            
        else:
            logger.warning("Missing vision_config, assuming defaults for head_size and linear_type")
            
    else:
        # No config provided, use defaults
        logger.warning("Missing model_config, assuming defaults for both language and vision models")


    # Get RoPE parameters for language model
    lang_rope_params = _get_rope_params(lang_linear_type)
    # Pattern for language model attention layers
    trans_required_pattern_lang = re.compile(
        f"language_model.base_model.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(lang_rope_params)})"
    )

    # Get RoPE parameters for vision model
    vision_rope_params = _get_rope_params(vision_linear_type)
    # Pattern for vision model attention layers
    trans_required_pattern_vision = re.compile(
        f"vision_tower.transformer.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(vision_rope_params)})"
    )

    
    # hf -> fms requires a transpose operation for the query and key
    # weight and bias parameters for Llama models
    # This transpose is due to the different implementation of RoPE in
    # HF and FMS. While FMS follows the original RoPE paper
    # (https://arxiv.org/abs/2104.09864), HF has its own implementation
    # that doesn't respect the order of outputs. This is OK as long as you
    # rearrange the weights of the query and key projections, as the
    # combination projection + RoPE ends up producing the same outputs.
    # Therefore, to make FMS produce the correct order of outputs when
    # loading from an HF checkpoint, we need to undo the transformation
    # that HF does from the original Meta weights:
    
    for name, param in input_sd.items():
        # Check if this parameter requires RoPE transformation for language model
        if trans_required_pattern_lang.match(name):
            # Apply RoPE transformation for language model
            new_sd[name] = _rope_transpose(param, lang_head_size, lang_linear_type )
        elif trans_required_pattern_vision.match(name):
            # Apply RoPE transformation for vision model
            param = _rope_transpose(param, vision_head_size, vision_linear_type)
        else:
            new_sd[name] = param

    return new_sd


def _rope_transpose(param: torch.Tensor, head_size: int, linear_type) -> torch.Tensor:
    """
    Transpose parameters for RoPE conversion between HF and FMS formats.
    
    Args:
        param: The parameter tensor to transpose
        head_size: Size of each attention head
        linear_type: linear type
    
    Returns:
        Transposed parameter tensor
    """
    if "gptq" in linear_type and param.dim() == 2:
        # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
        # and are fully transposed before & after process
        param = param.transpose(0, 1)

    # num_heads is used in the transformation required for hf->fms
    # can't be precomputed because q and k might have different num_heads
    num_heads = param.size(0) // head_size

    if param.dim() == 2:  # weight
        param_view = param.view(num_heads, 2, -1, param.size(1))
    else:  # bias
        param_view = param.view(num_heads, 2, -1)
    param = param_view.transpose(1, 2).reshape(*param.size())

    if "gptq" in linear_type and param.dim() == 2:
        param = param.transpose(0, 1)
    return param

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

def _weight_fusion(
    input_sd: Mapping, model_config: Optional[Mistral3Config] = None, **kwargs
):
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        new_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
            serialization._attn_unfused_to_fused_step(new_sd)
        )
    return new_sd

serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        # Language Model 
        (r"^language_model.lm_head.weight", "language_model.head.weight"),
        (r"^language_model.model.embed_tokens.weight", "language_model.base_model.embedding.weight"),
        (r"^language_model.model.norm", "language_model.base_model.dec_norm"),
        (r"^language_model.model.layers", "language_model.base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),

        # Vision Model 
        (r"feed_forward\.gate_proj", "ff_sub_layer.wg"),
        (r"feed_forward\.up_proj", "ff_sub_layer.w1"),
        (r"feed_forward\.down_proj","ff_sub_layer.w2"),
        (r"attention\.k_proj", "attn.in_proj.key"),
        (r"attention\.v_proj", "attn.in_proj.value"),
        (r"attention\.q_proj", "attn.in_proj.query"),
        (r"attention\.o_proj", "attn.dense"),   


    ]
    new_sd = {}
    for name, param in input_sd.items():
        # print("name:",name)
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "weight_fusion"],
)
