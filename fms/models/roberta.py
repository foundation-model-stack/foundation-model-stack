import math
import re
from dataclasses import dataclass
from typing import Optional, OrderedDict, Mapping, Any

import torch
import torch.nn as nn

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms import models
from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import FeedForwardBlock
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig
from fms.utils import serialization
from fms.modules.head import ClassificationHead


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
        )

        self.ff_sub_layer = FeedForwardBlock(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=True,
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
            k=x,
            v=x,
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

        # RoBERTa embeddings don't support TP as in many cases, the vocab size is not divisible by the world size
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

    def reset_params(self):
        for layer in ["embedding", "position_embedding"]:
            nn.init.normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=self.config.emb_dim**-0.5,
            )

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

        super(RoBERTa, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = RoBERTaConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = RoBERTaHeadless(self.config, self.distributed_strategy)

        # The head does not get TP-Wrapped as in many cases the vocab_size will not be divisible by the world size
        self.classification_head = self.distributed_strategy.distribute_module(
            ClassificationHead(
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
            self.classification_head.head.weight = self.base_model.embedding.weight

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

    def reset_params(self):
        self.base_model.reset_params()
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


# a micro llama model to use with a char-level tokenizer
_micro_char_config = RoBERTaConfig(
    emb_dim=192, nheads=4, nlayers=5, max_pos=1024, src_vocab_size=256
)

_base_config = RoBERTaConfig()

_architecture_name = "roberta"


def _roberta_factory_factory(config):
    def factory(**kwargs):
        return RoBERTa(config, **kwargs)

    return factory


models.register_model(
    _architecture_name, "micro", _roberta_factory_factory(_micro_char_config)
)
models.register_model(
    _architecture_name, "base", _roberta_factory_factory(_base_config)
)


def _hf_sd_to_fms_sd(hf_sd: Mapping[Any, Any]) -> Mapping[Any, Any]:
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
        (r"attention\.self\.key", "attn.key"),
        (r"attention\.self\.value", "attn.value"),
        (r"attention\.self\.query", "attn.query"),
        (r"attention\.output\.dense", "attn.dense"),
        (r"intermediate\.dense", "ff_sub_layer.w1"),
        (r"output\.dense", "ff_sub_layer.w2"),
        (r"^lm_head\.dense", "classification_head.dense"),
        (r"^lm_head\.layer_norm", "classification_head.ln"),
        (r"^lm_head\.decoder", "classification_head.head"),
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


serialization.register_adapter("roberta", "hf", _hf_sd_to_fms_sd)
