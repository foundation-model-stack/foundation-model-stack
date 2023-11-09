import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import FeedForwardBlock
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


@dataclass
class RoBERTaConfig(ModelConfig):
    src_vocab_size: int = 50265
    emb_dim: int = 768
    nheads: int = 12
    nlayers: int = 12
    pad_id: int = 1
    hidden_grow_factor: float = 4.0
    activation_fn: str = "gelu"
    max_pos: int = 512
    p_dropout: float = 0.1
    multiquery_attn: bool = False
    norm_eps: float = 1e-5


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
        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
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

        # then we do FF and Add&Norm
        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)

        if self.config.p_dropout != 0:
            x = self.dropout(x)

        # another residual
        x = x + residual

        return x


class RoBERTaHeadless(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [RoBERTaBlock(self.config) for _ in range(self.config.nlayers)]
        )

        self.embedding = nn.Embedding(self.config.src_vocab_size, self.config.emb_dim)

        self.enc_norm = nn.LayerNorm(self.config.emb_dim, eps=self.config.norm_eps)

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(self, x: torch.LongTensor, mask: Optional[torch.Tensor] = None):

        if mask is None:
            if x is None:
                raise ValueError("cannot create a mask when x is None")
            pad_id: int = self.config.pad_id
            is_pad: torch.BoolTensor = x == pad_id
            mask: torch.BoolTensor = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)

        x = self.embedding(x)

        # layer norm
        x = self.enc_norm(x)

        # add dropout
        if self.config.p_dropout:
            x = self.dropout(x)

        # layers
        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


class RoBERTaClassHead(nn.Module):
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.emb_dim, config.emb_dim)
        self.act = str_to_activation(config.activation_fn)
        self.ln = nn.LayerNorm(config.emb_dim, config.norm_eps)

    def forward(self, x: torch.FloatTensor):
        x = self.dense(x)
        x = self.act(x)
        return self.ln(x)


class RoBERTa(nn.Module):
    def __init__(self, config: Optional[RoBERTaConfig] = None, **kwargs):

        super(RoBERTa, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = RoBERTaConfig()
        self.config = self.config.updated(**kwargs)

        self.base_model = RoBERTaHeadless(config)

        self.class_head = RoBERTaClassHead(self.config)

        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

        # this model ties weights, so we tie here
        self.head.weight = self.base_model.embedding.weight

    def forward(self, x: torch.LongTensor, mask: Optional[torch.Tensor] = None):
        # run through the encoder layers
        x = self.base_model(x, mask=mask)

        # run through the class head (using the first in each sequence in the batch as the cls_token)
        x = self.class_head(x[:, 0, :])

        # project to vocab space
        x = self.lm_head(x)
        return x

    @classmethod
    def from_config(cls, config: RoBERTaConfig) -> "RoBERTa":
        return cls(config)

    def get_config(self) -> RoBERTaConfig:
        return self.config

    def reset_params(self):
        # Modules are self-initializing, we're just going to down-scale the final prediction head to be
        # mixed-fan (inputs and gradients scale to the same inverse factors) if it isn't tied
        self.head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * self.config.src_vocab_size)),
        )
