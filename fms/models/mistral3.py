import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple
from typing_extensions import Unpack

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import (
    DistributedStrategy,
    NoOpStrategy,
)

from fms.modules.attention import (
    AttentionKwargs,
    get_attention_type,
)

from fms.utils import serialization
from fms.utils.config import ModelConfig
from fms.utils.headless import gather_outputs

from fms.models.pixtral import PixtralVisionConfig
from fms.models.mistral import MistralConfig, MistralHeadless


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
    tie_heads: bool = False

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
            "text_config": self.text_config.to_dict()
            if hasattr(self.text_config, "to_dict")
            else vars(self.text_config),
            "vision_config": self.vision_config.to_dict()
            if hasattr(self.vision_config, "to_dict")
            else vars(self.vision_config),
            "projector_hidden_act": self.projector_hidden_act,
            "multimodal_projector_bias": self.multimodal_projector_bias,
            "spatial_merge_size": self.spatial_merge_size,
            "image_token_index": self.image_token_index,
            "vision_feature_layer": self.vision_feature_layer,
            "fused_weights": self.fused_weights,
        }
        return base


_24b_config = Mistral3Config()


class Mistral3(nn.Module):
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
        self.config.text_config = self.config.text_config.updated(**kwargs)
        self.config.vision_config = self.config.vision_config.updated(**kwargs)

        self.distributed_strategy = distributed_strategy

        self.base_model = MistralHeadless(
            self.config.text_config, self.distributed_strategy
        )
        self.head = nn.Linear(
            self.config.text_config.emb_dim,
            self.config.text_config.src_vocab_size,
            bias=False,
        )

    @classmethod
    def from_config(cls, config: Mistral3Config) -> "Mistral3":
        return cls(config)

    def get_config(self) -> ModelConfig:
        return self.config.text_config

    def reset_parameters(self):
        self.head.weight.data.normal_(
            0,
            1
            / math.sqrt(
                math.sqrt(
                    self.config.text_config.emb_dim
                    * self.config.text_config.src_vocab_size
                )
            ),
        )
        self.base_model.reset_parameters()

    def post_init(self):
        # if this model ties weights, they are tied here
        if self.config.text_config.tie_heads:
            # handle assignment of non-meta weights to meta parameters
            if self.head.weight.device == torch.device("meta"):
                self.head.weight = self.base_model.embedding.weight
            else:
                self.base_model.embedding.weight = self.head.weight

        self.base_model.post_init()

    def forward(
        self,
        x: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: bool = False,
        last_n_tokens: int = 0,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        get_attention_type(**attn_kwargs)["validate_attn_kwargs"](
            input_ids=x,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            **attn_kwargs,
        )
        output, cache = self.base_model(
            x, position_ids, past_key_value_states, use_cache, **attn_kwargs
        )

        output = gather_outputs(output, last_n_tokens, **attn_kwargs)
        preds = self.head(output)

        if use_cache:
            return preds, cache
        else:
            return preds


_architecture_name = "mistral3"


def _mistral3_factory_factory(config):
    def factory(**kwargs):
        return Mistral3(config, **kwargs)

    return factory


models.register_model(_architecture_name, "24b", _mistral3_factory_factory(_24b_config))


# =============== Serialization ==================


serialization.register_adapter_step(
    _architecture_name,
    "swiglu_unfused_to_fused",
    serialization._mlp_glu_unfused_to_fused_adapter_step,
)


def _weight_fusion(
    input_sd: Mapping[str, Any], model_config: Optional[Mistral3Config] = None, **kwargs
) -> Mapping[str, Any]:
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


def _hf_gptq_mistral3_check(
    input_sd: Mapping[str, Any], model_config: Optional[MistralConfig] = None, **kwargs
) -> Mapping[str, Any]:
    model_config = model_config.text_config  # type: ignore[union-attr]
    has_fused_weights = True
    linear_type = "torch_linear"
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]

    if "gptq" in linear_type and has_fused_weights:
        raise ValueError(
            "GPTQ HF mistral3 checkpoints cannot be loaded into a model with fused weights"
        )

    return input_sd


serialization.register_adapter_step(
    _architecture_name, "hf_gptq_fusion_check", _hf_gptq_mistral3_check
)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = replacements = [
        # Language Model
        (r"^language_model.lm_head.weight", "head.weight"),
        (r"^language_model.model.embed_tokens.weight", "base_model.embedding.weight"),
        (r"^language_model.model.norm", "base_model.dec_norm"),
        (r"^language_model.model.layers", "base_model.layers"),
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
        (r"feed_forward\.down_proj", "ff_sub_layer.w2"),
        (r"feed_forward\.down_proj", "ff_sub_layer.w2"),
        (r"attention\.k_proj", "attn.in_proj.key"),
        (r"attention\.v_proj", "attn.in_proj.value"),
        (r"attention\.q_proj", "attn.in_proj.query"),
        (r"attention\.o_proj", "attn.dense"),
        (r"attention\.o_proj", "attn.dense"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)


def _get_rope_params(linear_type: str) -> list[str]:
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[MistralConfig] = None, **kwargs
) -> Mapping[str, Any]:
    new_sd = {}
    model_config = model_config.text_config  # type: ignore[union-attr]
    if model_config:
        head_size = model_config.head_dim
        linear_type = "torch_linear"
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]
    else:
        logger.warning("Missing model_config, assuming defaults for head_size")
        head_size = 128  # Good default for most models
        linear_type = "torch_linear"

    rope_params = _get_rope_params(linear_type)
    trans_required_pattern = re.compile(
        f"base_model.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})"
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
        # that HF does from the original Meta weights:
        if bool(trans_required_pattern.search(name)):
            temp = param
            if "gptq" in linear_type and temp.dim() == 2:
                # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
                # and are fully transposed before & after process
                temp = temp.transpose(0, 1)
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # bias
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            if "gptq" in linear_type and temp.dim() == 2:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "hf_gptq_fusion_check", "weight_fusion"],
)
