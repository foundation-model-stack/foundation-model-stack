import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from fms.utils.config import ModelConfig
from fms.modules.feedforward import FeedForwardBlock
from fms.utils.activation import str_to_activation

from fms.modules.attention import MultiHeadAttention


@dataclass
class GPTBigCodeConfig(ModelConfig):
    src_vocab_size: int = 49157  # This param default is based on https://huggingface.co/bigcode/gpt_bigcode-santacoder
    emb_dim: int = 2048  # This param default is based on https://huggingface.co/bigcode/gpt_bigcode-santacoder
    nheads: int = 12
    nlayers: int = 12
    pad_id: int = 0
    max_pos: int = 512
    hidden_grow_factor: float = 4.0
    activation_fn: str = "gelu-tanh"
    p_dropout: float = 0.0
    emb_dropout: float = 0.0
    multiquery_attn: bool = True
    ln_eps: float = 1e-5


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        super().__init__()
        self.config = config

        self.ln = nn.LayerNorm(self.config.emb_dim, self.config.ln_eps)
        self.ff_ln = nn.LayerNorm(self.config.emb_dim, self.config.ln_eps)

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
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_state: Optional[
            Tuple[
                torch.Tensor,
            ]
        ] = None,
        use_cache: bool = False,
        is_causal_mask: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        self_attn_past_key_value = past_key_value_state

        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
        # self attention
        x = self.attn(
            q=x,
            k=x,
            v=x,
            mask=mask,
            position_ids=position_ids,
            attn_algorithm=attn_algorithm,
            past_key_value_state=self_attn_past_key_value,
            use_cache=use_cache,
            is_self=True,
            is_causal_mask=is_causal_mask,
        )

        cache = None
        if use_cache:
            x, cache = x
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

        if use_cache:
            return x, cache
        else:
            return x


class GPTBigCodeHeadless(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            [GPTBigCodeBlock(self.config) for _ in range(self.config.nlayers)]
        )

        self.embedding = nn.Embedding(self.config.src_vocab_size, self.config.emb_dim)
        self.position_embedding = nn.Embedding(self.config.max_pos, self.config.emb_dim)

        self.dec_norm = nn.LayerNorm(self.config.emb_dim, eps=self.config.ln_eps)

        if self.config.emb_dropout:
            self.emb_dropout = nn.Dropout(self.config.emb_dropout)

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def _compute_position_ids(
        self,
        is_pad: torch.Tensor,
        use_cache: bool,
        past_key_value_states: Optional[
            List[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]
        ] = None,
    ):
        """compute the position ids if the use happened not to give any"""
        position_ids = ((~is_pad).cumsum(1) - 1).clamp(min=0)

        # Compute position_ids based on cache config
        if (
            use_cache
            and past_key_value_states is not None
            and past_key_value_states[0] is not None
        ):
            position_ids += past_key_value_states[0][0].size(-2)

        return position_ids

    def forward(
        self,
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[
            List[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]
        ] = None,
        use_cache: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len

        qlen = x.size(1)
        klen = x.size(1)

        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        # if we are using the cache, the key length needs to be extended with the past keys length
        if (
            use_cache
            and past_key_value_states is not None
            and past_key_value_states[0] is not None
        ):
            klen += past_key_value_states[0][0].size(-2)

        # if mask is none, we need to compute mask
        is_causal_mask = False
        if mask is None:
            if x is None:
                raise ValueError("cannot create a mask when x is None")
            # we are caching and can assume all 1s in the mask
            if use_cache and klen != 1 and qlen == 1:
                # b x h x qlen x kvlen
                mask = torch.ones(qlen, klen, device=x.device)
            else:
                pad_id: int = self.config.pad_id
                is_pad: torch.Tensor = x == pad_id
                mask = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
                mask = mask.tril(diagonal=0)

        x_emb = self.embedding(x)

        # if pad_id exists
        #   is_pad will be a BoolTensor
        #   otherwise pad_id will not be taken into account
        if self.config.pad_id is None:
            is_pad = torch.zeros_like(x, dtype=bool, device=x.device)
        else:
            is_pad = x == self.config.pad_id

        if position_ids is None:
            position_ids = self._compute_position_ids(
                is_pad, use_cache, past_key_value_states
            )

        # look up position embeddings
        position_out = self.position_embedding(position_ids)

        # zero out the associated position embeddings
        if self.config.pad_id is not None:
            position_out = position_out.mul(~is_pad.unsqueeze(-1))

        # perform absolute position embedding
        x = x_emb + position_out

        # apply dropout to embeddings
        if self.config.emb_dropout:
            x = self.emb_dropout(x)

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x,
                mask=mask,
                is_causal_mask=is_causal_mask,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                attn_algorithm=attn_algorithm,
            )

            if use_cache:
                x, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)

            else:
                x = output

        dec_out = self.dec_norm(x)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


# Implements the decoder-only GPTBigCodeModel
class GPTBigCode(nn.Module):
    def __init__(
        self,
        config: Optional[GPTBigCodeConfig] = None,
        **kwargs,
    ):
        super(GPTBigCode, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = GPTBigCodeConfig()
        self.config = self.config.updated(**kwargs)

        self.base_model = GPTBigCodeHeadless(self.config)
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

        # this model ties weights, so we tie here
        self.head.weight = self.base_model.embedding.weight

        self.reset_params()

    @classmethod
    def from_config(cls, config: GPTBigCodeConfig) -> "GPTBigCode":
        return cls(config)

    def get_config(self) -> GPTBigCodeConfig:
        return self.config

    def reset_params(self):
        # Modules are self-initializing, we're just going to down-scale the final prediction head to be
        # mixed-fan (inputs and gradients scale to the same inverse factors) if it isn't tied
        self.head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * self.config.src_vocab_size)),
        )

    def forward(
        self,
        x: torch.LongTensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[
            Tuple[
                torch.FloatTensor,
            ]
        ] = None,
        use_cache: bool = False,
        attn_algorithm: Optional[str] = None,
    ):
        output, cache = self.base_model(
            x,
            mask,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            use_cache=use_cache,
            attn_algorithm=attn_algorithm,
        )

        preds = self.head(output)

        if use_cache:
            return preds, cache
        else:
            return preds
