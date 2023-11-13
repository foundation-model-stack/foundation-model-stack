import math
import re
from dataclasses import dataclass
from typing import Optional

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
        x: torch.LongTensor,
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

        self.layers = nn.ModuleList(
            [
                distributed_strategy.distribute_layer(RoBERTaBlock(self.config), i)
                for i in range(self.config.nlayers)
            ]
        )

        self.embedding = nn.Embedding(self.config.src_vocab_size, self.config.emb_dim)

        self.position_embedding = nn.Embedding(self.config.max_pos, self.config.emb_dim)

        self.enc_norm = distributed_strategy.distribute_module(
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
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attn_algorithm: Optional[str] = None,
    ):

        if mask is None:
            if x is None:
                raise ValueError("cannot create a mask when x is None")
            pad_id: int = self.config.pad_id
            is_pad: torch.BoolTensor = x == pad_id
            mask: torch.BoolTensor = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)

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

        self.base_model = RoBERTaHeadless(self.config, distributed_strategy)

        self.classification_head = ClassificationHead(
            self.config.emb_dim,
            # number of classes is vocab size as this is predicting a masked token
            num_classes=self.config.src_vocab_size,
            activation_fn=str_to_activation(self.config.activation_fn),
            layer_norm=nn.LayerNorm(self.config.emb_dim, self.config.norm_eps),
            dropout=self.config.p_dropout,
        )

        # this model ties weights, so we tie here
        if self.config.tie_heads:
            self.classification_head.head.weight = self.base_model.embedding.weight

    def forward(
        self,
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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


def _hf_sd_to_fms_sd(hf_sd):
    result = {}

    # process embeddings
    result["base_model.embedding.weight"] = hf_sd[
        "roberta.embeddings.word_embeddings.weight"
    ]
    result["base_model.position_embedding.weight"] = hf_sd[
        "roberta.embeddings.position_embeddings.weight"
    ][2:]

    def _apply_weight_bias(hf_value, fms_value):
        result[f"{fms_value}.weight"] = hf_sd[f"{hf_value}.weight"]
        result[f"{fms_value}.bias"] = hf_sd[f"{hf_value}.bias"]

    # process layers
    layer_pattern = re.compile("roberta.encoder.layer.[0-9]+")
    processed_layers = set()
    for hf_k, hf_v in hf_sd.items():
        match = layer_pattern.match(hf_k)
        if bool(match):
            layer = f"{hf_k[: match.regs[0][1]]}"

            # only process the layer if we have not seen it yet
            if layer not in processed_layers:
                layer_i = re.search("\d+|$", layer).group()
                fms_layer = f"base_model.layers.{layer_i}"

                # layer norm
                _apply_weight_bias(
                    f"{layer}.attention.output.LayerNorm", f"{fms_layer}.ln"
                )
                _apply_weight_bias(f"{layer}.output.LayerNorm", f"{fms_layer}.ff_ln")

                # attn
                _apply_weight_bias(
                    f"{layer}.attention.self.query", f"{fms_layer}.attn.query"
                )
                _apply_weight_bias(
                    f"{layer}.attention.self.key", f"{fms_layer}.attn.key"
                )
                _apply_weight_bias(
                    f"{layer}.attention.self.value", f"{fms_layer}.attn.value"
                )
                _apply_weight_bias(
                    f"{layer}.attention.output.dense", f"{fms_layer}.attn.dense"
                )

                # ff
                _apply_weight_bias(
                    f"{layer}.intermediate.dense", f"{fms_layer}.ff_sub_layer.w1"
                )
                _apply_weight_bias(
                    f"{layer}.output.dense", f"{fms_layer}.ff_sub_layer.w2"
                )

                processed_layers.add(layer)

    # process model layer norm
    _apply_weight_bias("roberta.embeddings.LayerNorm", "base_model.enc_norm")

    # process model head
    if (
        "lm_head.dense.weight" in hf_sd
        and "lm_head.layer_norm.weight" in hf_sd
        and "lm_head.decoder.bias" in hf_sd
    ):
        _apply_weight_bias("lm_head.dense", "classification_head.dense")
        _apply_weight_bias("lm_head.layer_norm", "classification_head.ln")
        _apply_weight_bias("lm_head.decoder", "classification_head.head")
    else:
        print(
            "This model does not have the default head, and therefore requires manual copying for the head"
        )

    return result


serialization.register_adapter("roberta", "hf", _hf_sd_to_fms_sd)
