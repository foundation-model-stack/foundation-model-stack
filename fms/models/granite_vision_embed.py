# /models/huggingface_cache/modules/transformers_modules/ibm-granite/granite-vision-3.3-2b-embedding/baa53d1e1ac95baefb15e4aa9fdcc471ac728c83
from dataclasses import dataclass
import logging
from typing import Any, Mapping, Optional, Unpack, Tuple
import re
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms import models
from fms.models.llava_next import LlavaNext, LlavaNextConfig
from fms.utils import serialization
from fms.modules.attention import AttentionKwargs
from fms.modules.linear import get_linear_type

import numpy as np
import torch
# Used in custom unpadding utils
from transformers.models.llava_next.modeling_llava_next import unpad_image, get_anyres_image_grid_shape

import torch
from torch import nn

logger = logging.getLogger(__name__)

@dataclass
class GraniteVisionEmbConfig(LlavaNextConfig):
    # TODO: abstract configs for underlying granite/siglip configs under
    # granite vision since we will use the same ones here, and should be
    # able to just pass the version through.
    emb_dim_query: int = 128
    emb_dim_doc: int = 128
    base_image_feature_location: str = "last"
    # NOTE: For now we exclude adapter_path since it's not actually used;
    # seems like maybe a LoRA adapter might have been initially present,
    # and merged with the weights?

# Granite vision embeddings are very similar to llava next / granite vision.
# TODO should be the 3.3 config, not 3.2, but that would go in the main vision file...
_granite_vision_embed_3_3_2b_config = GraniteVisionEmbConfig()
_architecture_name = "granite_vision_emb"

def _granite_vision_embed_factory_factory(config):
    def factory(**kwargs):
        return GraniteVisionEmb(config, **kwargs)

    return factory

# TODO - add this
# TODO add custom packing
# TODO learn to run deepview
class GraniteVisionEmb(nn.Module):
    def __init__(
        self,
        config: Optional[LlavaNextConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(GraniteVisionEmb, self).__init__()
        # Since the config is identical to Llava Next, we leave the default
        # handling to that class, but we go with encapsulation instead of
        # inheritance here to better align with the model structure on HF hub.
        self.model = LlavaNext(config, distributed_strategy, **kwargs)

        # TODO - projector dim is currently hardcoded to 128 in the module,
        # but it's probably a good idea to pull that out to a config on HF Hub
        # and pull it out of the config here...
        self.custom_text_proj = nn.Linear(
            self.model.config.text_config.emb_dim, 128,
        )

        # NOTE - this attr is needed to pass this as an adapter kwarg
        self.config = self.model.config
        self.distributed_strategy = distributed_strategy

    def prepare_inputs_for_generation(
        self,
        iteration,
        input_ids,
        kwargs,
    ):
        """Extracts the image features and creates the multimodal inputs.
        NOTE: We should rework this interface to make sense for embed,
        because we don't really need a prefill type operation for running
        the vision tower since there is only forward call / iteration anyway.
        However, for now we right it this way so that the usage matches
        generate(), and the pattern aligns with llava next.
        """
        inputs = kwargs.get("inputs")

        pixel_values = kwargs.get("pixel_values")
        image_sizes = kwargs.get("image_sizes")

        # No image data to pre-process
        if pixel_values is None or pixel_values.size(0) == 0:
            return input_ids, kwargs

        if input_ids is None and inputs is None:
            # Image token is still needed for image encoding, so
            # technically we always need to have text, even for just
            # encoding images.
            raise ValueError("input_ids and inputs can't both be None")

        # embedded inputs supersede input_ids; note the extra
        # layer of wrapping here when compared to llava next.
        if inputs is None:
            inputs = self.model.language_model.base_model.embedding(input_ids)

        # Retrieving the visual encoder outputs and repack
        image_features = self.get_image_features(
            pixel_values,
            image_sizes,
        )

        image_features, _ = self.pack_image_features(
            image_features,
            image_sizes,
            image_newline=self.model.image_newline,
        )

        squeezed_image_mask = (input_ids == self.config.image_token_index)

        # Rescatter the image features into the corresponding <image> features indices
        special_image_mask = squeezed_image_mask.unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs).to(inputs.device)
        image_features = image_features.to(inputs.device, inputs.dtype)

        inputs = inputs.masked_scatter(special_image_mask, image_features)

        # HACK get the last_k_indices here...
        last_k_indices = self.get_topk_indices(squeezed_image_mask)
        kwargs["last_k_indices"] = last_k_indices
        return inputs, kwargs

    def get_topk_indices(self, image_mask):
        N, M = image_mask.shape # We should have a 2D mask here (not unsqueezed)

        # Create an index matrix: each row is 0, 1, ..., M-1
        idx = torch.arange(M, device=image_mask.device).expand(N, M)
        # Replace False positions with -1 so they are ignored by topk (since all valid indices are >=0)
        masked_idx = torch.where(image_mask, idx, torch.tensor(-1, device=image_mask.device))
        topk_values, _ = torch.topk(masked_idx, k=729, dim=1)
        last_k_indices, _ = torch.sort(topk_values, dim=1)
        return last_k_indices


    # HF impl - copy paste from llava next, we should share code more cleanly
    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ):
        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            self.model.image_size_to_num_patches( # HACK - added extra nest level
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]

        if pixel_values.dim() == 5:
            # stacked if input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [
                pix_val[:num_patch]
                for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(
                f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions"
            )


        _, _, image_features = self.model.vision_tower( #FIXME - extra layer of wrapping
            pixel_values,
            output_hidden_states=True,
        )

        if isinstance(self.config.vision_feature_layer, int):
            selected_image_feature = image_features[self.config.vision_feature_layer]
        else:
            hs_pool = [
                image_features[layer_idx]
                for layer_idx in self.config.vision_feature_layer
            ]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        if self.config.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]

        image_features = self.model.multi_modal_projector(selected_image_feature) # FIXME extra layer of wrapping
        image_features = torch.split(image_features, image_num_patches, dim=0)
        return image_features

    def post_init(self):
        # Post init of the granite vision VLM encapsulates
        # post init calls for both the visual encoder and LLM.
        self.model.post_init()

    def pack_image_features(
            self,
            image_features,
            image_sizes,
            # vision feature select is always all for siglip (no CLS)
            image_newline=None
    ):
        """
        # NOTE - currently a copy/paste from the original code to see what breaks.
        Would be best to align this with the llava next implementation.

        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.
        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            image_sizes (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        base_image_feature_location = self.config.base_image_feature_location
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )

                # if (
                #         np.prod(image_feature.shape) % (num_patch_height * num_patch_width * height * width) != 0
                #         and vision_feature_select_strategy == "default"
                # ):
                #     print(
                #         "Image feature shape does not line up with the provided patch size. "
                #         "You may be using the `default` vision_feature_select_strategy with a"
                #         " visual encoder that does not have CLS."
                #     )

                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                if base_image_feature_location == "last":
                    image_feature = torch.cat((image_feature, base_image_feature), dim=0)
                else:
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)

            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens



    def forward(
        self,
        input_ids_or_embeds: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: Optional[bool] = False,
        last_k_indices = None,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        if use_cache: # only kept for compatability
            logger.warning("Setting use_cache to False for embedding models")
            use_cache = False

        # NOTE - this is the same as calling the underlying llava next model,
        # but we directly call the LLM here to make it more obvious what is
        # happening; the preparation of the image inputs takes place in
        # prepare_inputs_for_generation to better align with llava next impl,
        # since image inputs only need to be masked into the inital embeddings.
        last_hidden_states = self.model.language_model(
            input_ids_or_embeds,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            use_cache=False,
            run_headless=True,
            **attn_kwargs,
        )

        # TODO - check masking correctness across different implementations besides SDPA
        attention_mask = attn_kwargs["attention_mask"] # [bsz, N]

        # Apply top k
        if last_k_indices is not None:
            last_k_indices_exp = last_k_indices.unsqueeze(-1).expand(-1, -1, last_hidden_states.size(-1))
            last_hidden_states = torch.gather(last_hidden_states, 1, last_k_indices_exp)
            attention_mask = torch.gather(attention_mask, 1, last_k_indices)

        attention_mask = attention_mask.unsqueeze(-1)

        # Get the last hidden states and apply the text projector
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-8)

        proj = proj  * attention_mask  # (batch_size, sequence_length, dim)
        return proj

models.register_model(
    _architecture_name,
    "_granite_vision_embed_3_3_2b_config",
    _granite_vision_embed_factory_factory(_granite_vision_embed_3_3_2b_config),
)

def _weight_fusion(
    input_sd: Mapping, model_config: Optional[LlavaNextConfig] = None, **kwargs
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


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    # TODO - we use the same replacement scheme as llava next, but it's good
    # to be aware that there is an extra layer of encapsulation here.
    replacements = [
        # vision
        (r"vision_tower\.vision_model\.head", "vision_tower.head"),
        (r"vision_tower\.vision_model\.encoder", "vision_tower.base_model.encoder"),
        (
            r"vision_tower\.vision_model\.embeddings",
            "vision_tower.base_model.embeddings",
        ),
        (
            r"vision_tower\.vision_model\.post_layernorm",
            "vision_tower.base_model.post_layernorm",
        ),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.out_proj", "attn.dense"),
        (r"mlp\.fc1", "mlp.w1"),
        (r"mlp\.fc2", "mlp.w2"),
        # language
        (r"language_model\.lm_head\.weight", "language_model.head.weight"),
        (
            r"language_model.model.embed_tokens.weight",
            "language_model.base_model.embedding.weight",
        ),
        (r"language_model.model.norm", "language_model.base_model.dec_norm"),
        (r"language_model.model.layers", "language_model.base_model.layers"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(f"{pattern}", repl, new_name)
        new_sd[new_name] = param
    return new_sd

# From Granite model
def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    if "int8" in linear_type:
        # quantize_weight is fms-model-optimizer identifier of weight clip values
        return ["weight", "bias", "quantize_weight"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


# From Granite model
def _hf_to_fms_rope(
    input_sd: Mapping[str, Any],
    model_config=None,
    **kwargs,
) -> Mapping[str, Any]:
    new_sd = {}
    if model_config:
        model_config = model_config.text_config

    if model_config:
        head_size = model_config.emb_dim // model_config.nheads
        linear_type_str = "torch_linear"
        if model_config.linear_config:
            linear_type_str = get_linear_type(
                model_config.linear_config,
                module_name=None,  # if callable, linear_type should return default str
            )
    else:
        logger.warning("Missing model_config, assuming defaults for head_size")
        head_size = 128  # Good default for most models
        linear_type_str = "torch_linear"

    rope_params = _get_rope_params(linear_type_str)
    # TODO - this is exactly the same as llava_next, but we prepended model. to capture correctly
    # we should refactor this into a common util and also warn / throw if SD entries are mapped
    # properly.
    trans_required_pattern = re.compile(
        f"model.language_model.base_model.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})"
    )
    for name, param in input_sd.items():
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
        # that HF does from the original Meta weights
        is_gptq_2d_qparam = "gptq" in linear_type_str and param.dim() == 2
        if bool(trans_required_pattern.match(name)) and param.numel() > 1:
            temp = param
            if is_gptq_2d_qparam:
                # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
                # and are fully transposed before & after process.
                # GPTQ scales and qzeros are also transposed accordingly
                temp = temp.transpose(0, 1)
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # 1-dim parameters
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            if is_gptq_2d_qparam:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "weight_fusion"],
)
