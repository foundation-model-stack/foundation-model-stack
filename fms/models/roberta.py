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
    classifier_activation_fn: str = "tanh"
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
    def __init__(self, config: RoBERTaConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [RoBERTaBlock(self.config) for _ in range(self.config.nlayers)]
        )

        self.embedding = nn.Embedding(self.config.src_vocab_size, self.config.emb_dim)
        self.position_embedding = nn.Embedding(self.config.max_pos, self.config.emb_dim)

        self.enc_norm = nn.LayerNorm(self.config.emb_dim, eps=self.config.norm_eps)

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
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
        x = self.ln(x)
        return x


class RoBERTa(nn.Module):
    def __init__(self, config: Optional[RoBERTaConfig] = None, **kwargs):

        super(RoBERTa, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = RoBERTaConfig()
        self.config = self.config.updated(**kwargs)

        self.base_model = RoBERTaHeadless(self.config)

        self.class_head = RoBERTaClassHead(self.config)

        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

        # this model ties weights, so we tie here
        self.head.weight = self.base_model.embedding.weight

    def forward(
        self,
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        # run through the encoder layers
        x = self.base_model(x, mask=mask, position_ids=position_ids)

        # run through the class head (using the first in each sequence in the batch as the cls_token)
        x = self.class_head(x)

        # project to vocab space
        x = self.head(x)
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
