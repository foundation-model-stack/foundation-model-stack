import copy
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import FeedForwardBlock
from fms.modules.head import MLPClassificationHead
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


@dataclass
class RoBERTaConfig(ModelConfig):
    src_vocab_size: int = 50265
    emb_dim: int = 768
    nheads: int = 12
    nlayers: int = 12
    pad_id: int = 1
    hidden_grow_factor: float = 4.0
    activation_fn: str = "gelu"
    classifier_activation_fn: str = "tanh"
    max_pos: int = 512
    p_dropout: float = 0.1
    multiquery_attn: bool = False
    norm_eps: float = 1e-12
    tie_heads: bool = False
    linear_config: Optional[Mapping[str, Any]] = None
    fused_weights: bool = True


@dataclass
class RoBERTaQuestionAnsweringConfig(RoBERTaConfig):
    """Model configuration of RoBERTa for Question-Answering downstream task"""

    num_classes: int = 2


class RoBERTaBlock(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.config = config

        self.ln = nn.LayerNorm(self.config.emb_dim, self.config.norm_eps)
        self.ff_ln = nn.LayerNorm(self.config.emb_dim, self.config.norm_eps)

        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            self.config.emb_dim // self.config.nheads,
            self.config.emb_dim // self.config.nheads,
            self.config.nheads,
            kvheads=1 if self.config.multiquery_attn else self.config.nheads,
            p_dropout=self.config.p_dropout,
            use_bias=True,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )

        self.ff_sub_layer = FeedForwardBlock(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=True,
            linear_config=self.config.linear_config,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        attn_algorithm: Optional[str] = None,
    ):
        # first we do MHA
        residual = x
        # self attention
        x = self.attn(
            q=x,
            mask=mask,
            attn_algorithm=attn_algorithm,
            is_self=True,
            is_causal_mask=False,
        )

        if self.config.p_dropout != 0:
            x = self.dropout(x)

        # residual connection
        x = x + residual
        # post ln
        x = self.ln(x)

        # then we do FF and Add&Norm
        residual = x
        x = self.ff_sub_layer(x)

        if self.config.p_dropout != 0:
            x = self.dropout(x)

        # another residual
        x = x + residual
        x = self.ff_ln(x)

        return x


class RoBERTaHeadless(nn.Module):
    def __init__(
        self, config: RoBERTaConfig, distributed_strategy: DistributedStrategy
    ):
        super().__init__()
        self.config = config
        self.distributed_strategy = distributed_strategy

        self.layers = nn.ModuleList(
            [
                self.distributed_strategy.distribute_layer(RoBERTaBlock(self.config), i)
                for i in range(self.config.nlayers)
            ]
        )

        # RoBERTa embeddings don't support TP as in many cases the vocab size is
        # not divisible by the world size
        self.embedding = self.distributed_strategy.distribute_module(
            nn.Embedding(self.config.src_vocab_size, self.config.emb_dim),
            final_layers=True,
        )

        self.position_embedding = self.distributed_strategy.distribute_module(
            nn.Embedding(self.config.max_pos, self.config.emb_dim),
            final_layers=True,
        )

        self.enc_norm = self.distributed_strategy.distribute_module(
            nn.LayerNorm(self.config.emb_dim, eps=self.config.norm_eps),
            final_layers=True,
        )

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def reset_parameters(self):
        for layer in ["embedding", "position_embedding"]:
            nn.init.normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=self.config.emb_dim**-0.5,
            )
        for layer in self.layers:
            for sublayer in ["ln", "ff_ln", "attn", "ff_sub_layer"]:
                getattr(layer, sublayer).reset_parameters()
        self.enc_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attn_algorithm: Optional[str] = None,
    ):
        if mask is None:
            if x is None:
                raise ValueError("cannot create a mask when x is None")
            pad_id: int = self.config.pad_id
            is_pad = x == pad_id
            mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)

        x_emb = self.embedding(x)

        # if pad_id exists
        #   is_pad will be a BoolTensor
        #   otherwise pad_id will not be taken into account
        if self.config.pad_id is None:
            is_pad = torch.zeros_like(x, dtype=bool, device=x.device)
        else:
            is_pad = x == self.config.pad_id

        if position_ids is None:
            position_ids = ((~is_pad).cumsum(1) - 1).clamp(min=0)

        # look up position embeddings
        position_out = self.position_embedding(position_ids)

        # zero out the associated position embeddings
        if self.config.pad_id is not None:
            position_out = position_out.mul(~is_pad.unsqueeze(-1))

        # perform absolute position embedding
        x = x_emb + position_out

        # layer norm
        x = self.enc_norm(x)

        # add dropout
        if self.config.p_dropout:
            x = self.dropout(x)

        # layers
        for layer in self.layers:
            x = layer(x, mask=mask, attn_algorithm=attn_algorithm)

        return x


class RoBERTa(nn.Module):
    def __init__(
        self,
        config: Optional[RoBERTaConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = RoBERTaConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = RoBERTaHeadless(self.config, self.distributed_strategy)

        # The head does not get TP-Wrapped as in many cases the vocab_size
        # will not be divisible by the world size
        self.classification_head = self.distributed_strategy.distribute_module(
            MLPClassificationHead(
                self.config.emb_dim,
                # number of classes is vocab size as this is predicting a masked token
                num_classes=self.config.src_vocab_size,
                activation_fn=str_to_activation(self.config.activation_fn),
                layer_norm=nn.LayerNorm(self.config.emb_dim, self.config.norm_eps),
                dropout=self.config.p_dropout,
            ),
            final_layers=True,
        )

        # this model ties weights, so we tie here
        if self.config.tie_heads:
            self.classification_head.get_submodule(
                "head"
            ).weight = self.base_model.embedding.weight

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attn_algorithm: Optional[str] = None,
    ):
        # run through the encoder layers
        x = self.base_model(
            x, mask=mask, position_ids=position_ids, attn_algorithm=attn_algorithm
        )

        # run through classification head and project to vocab space
        x = self.classification_head(x)
        return x

    @classmethod
    def from_config(cls, config: RoBERTaConfig) -> "RoBERTa":
        return cls(config)

    def get_config(self) -> RoBERTaConfig:
        return self.config

    def reset_parameters(self):
        self.base_model.reset_parameters()
        if self.config.tie_heads:
            self.classification_head.head.bias.data.zero_()
        else:
            self.classification_head.head.weight.data.normal_(
                0,
                1
                / math.sqrt(
                    math.sqrt(self.config.emb_dim * self.config.src_vocab_size)
                ),
            )

    def post_init(self):
        # This function is called in `get_model` after the model is fully initalized
        # on the correct device

        # if this model ties weights, so we tie here
        if self.config.tie_heads:
            # make sure you assign the non-meta weights to the meta parameter
            if self.classification_head.head.weight.device == torch.device("meta"):
                self.classification_head.head.weight = self.base_model.embedding.weight
            else:
                self.base_model.embedding.weight = self.classification_head.head.weight


class RoBERTaForQuestionAnswering(nn.Module):
    """Model architecture of RoBERTa for Question Answering downstream task"""

    def __init__(
        self,
        config: Optional[RoBERTaQuestionAnsweringConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            self.config = RoBERTaQuestionAnsweringConfig()
        self.config = self.config.updated(**kwargs)
        if self.config.tie_heads:
            logger.warning(
                "The model configuration set tie heads to True but this parameter will "
                "be ignored for a QuestionAnswering task."
            )
        self.distributed_strategy = distributed_strategy

        self.base_model = RoBERTaHeadless(self.config, self.distributed_strategy)

        # The head does not get TP-wrapped and is not quantized
        # output dimension ("num_classes") for QuestionAnswering is always 2
        self.qa_head = nn.Linear(
            in_features=self.config.emb_dim,
            out_features=2,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attn_algorithm: Optional[str] = None,
    ):
        # run through the encoder layers
        x = self.base_model(
            x, mask=mask, position_ids=position_ids, attn_algorithm=attn_algorithm
        )

        # run head and process outputs
        logits = self.qa_head(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        return (start_logits, end_logits)

    @classmethod
    def from_config(
        cls, config: RoBERTaQuestionAnsweringConfig
    ) -> "RoBERTaForQuestionAnswering":
        return cls(config)

    def get_config(self) -> RoBERTaQuestionAnsweringConfig:
        return self.config

    def reset_parameters(self):
        self.base_model.reset_parameters()
        self.qa_head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * 2)),
        )


# a micro llama model to use with a char-level tokenizer
_micro_char_config = RoBERTaConfig(
    emb_dim=192, nheads=4, nlayers=5, max_pos=1024, src_vocab_size=256
)

_base_config = RoBERTaConfig(tie_heads=True, norm_eps=1e-5, p_dropout=0.1)

_base_questionanswering_config_dict = copy.copy(_base_config.__dict__)
_base_questionanswering_config_dict["tie_heads"] = False
_base_questionanswering_config = RoBERTaQuestionAnsweringConfig(
    **_base_questionanswering_config_dict,
    num_classes=2,
)

_architecture_name = "roberta"


def _roberta_factory_factory(config):
    def factory(**kwargs):
        return RoBERTa(config, **kwargs)

    return factory


def _roberta_question_answering_factory_factory(config):
    def factory(**kwargs):
        return RoBERTaForQuestionAnswering(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "micro", _roberta_factory_factory(_micro_char_config)
)
models.register_model(
    _architecture_name, "base", _roberta_factory_factory(_base_config)
)
models.register_model(
    "roberta_question_answering",
    "base",
    _roberta_question_answering_factory_factory(_base_questionanswering_config),
)

serialization.register_adapter_step(
    _architecture_name,
    "pre0.0.6_attn_unfused_to_fused",
    serialization._pre006_attn_adapter_step,
)


def _weight_fusion(
    input_sd: Mapping[str, Any], model_config: Optional[RoBERTaConfig] = None, **kwargs
) -> Mapping[str, Any]:
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        new_sd = serialization._attn_unfused_to_fused_step(new_sd)
    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)
serialization.register_adapter_step(
    "roberta_question_answering",
    "weight_fusion",
    _weight_fusion,
)


def _hf_to_fms_names(hf_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    replacements = [
        (r"^roberta.embeddings.word_embeddings.weight", "base_model.embedding.weight"),
        (
            r"^roberta.embeddings.position_embeddings.weight",
            "base_model.position_embedding.weight",
        ),
        (r"^roberta.embeddings.LayerNorm", "base_model.enc_norm"),
        (r"^roberta.encoder.layer", "base_model.layers"),
        (r"attention\.output\.LayerNorm", "ln"),
        (r"output\.LayerNorm", "ff_ln"),
        (r"attention\.self\.key", "attn.in_proj.key"),
        (r"attention\.self\.value", "attn.in_proj.value"),
        (r"attention\.self\.query", "attn.in_proj.query"),
        (r"attention\.output\.dense", "attn.dense"),
        (r"intermediate\.dense", "ff_sub_layer.w1"),
        (r"output\.dense", "ff_sub_layer.w2"),
        (r"^lm_head\.dense", "classification_head.dense"),
        (r"^lm_head\.layer_norm", "classification_head.ln"),
        (r"^lm_head\.decoder", "classification_head.head"),
        (r"^qa_outputs", "qa_head"),  # only relevant to QuestionAnswering task
    ]
    new_sd = {}
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

        # hf always has the first 2 spots set, we need to remove them as they are not used
        if name == "roberta.embeddings.position_embeddings.weight":
            new_sd[new_name] = new_sd[new_name][2:]

    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)
serialization.register_adapter_step(
    "roberta_question_answering", "hf_to_fms_names", _hf_to_fms_names
)

serialization.register_adapter("roberta", "hf", ["hf_to_fms_names", "weight_fusion"])
serialization.register_adapter(
    "roberta", "fms.pre0.0.6", ["pre0.0.6_attn_unfused_to_fused", "weight_fusion"]
)
serialization.register_adapter(
    "roberta_question_answering", "hf", ["hf_to_fms_names", "weight_fusion"]
)
