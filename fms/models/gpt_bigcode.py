import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from fms.utils.config import ModelConfig
from fms.modules.embedding import AbsolutePositionEmbedding
from fms.modules.feedforward import FeedForwardBlock
from fms.utils.activation import str_to_activation

from fms.modules.attention import MultiHeadAttention


@dataclass
class GPTBigCodeConfig(ModelConfig):
    src_vocab_size: int = 49280
    emb_dim: int = 2048
    emb_kq: Optional[int] = None
    emb_v: Optional[int] = None
    nheads: int = 12
    kvheads: int = 1
    nlayers: int = 12
    pad_id: int = 0
    max_pos: int = 512
    vocab_bias: bool = False
    use_bias: bool = True
    hidden_grow_factor: float = 4.0
    activation_fn: str = "gelu-tanh"
    p_dropout: float = 0.0
    emb_dropout: float = 0.0
    ln_eps: float = 1e-5


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config: GPTBigCodeConfig):
        super().__init__()
        self.config = config

        self.ln = nn.LayerNorm(self.config.emb_dim, self.config.ln_eps)
        self.ff_ln = nn.LayerNorm(self.config.emb_dim, self.config.ln_eps)

        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads
        if self.config.kvheads == 0:
            kvheads = self.config.nheads
        else:
            kvheads = self.config.kvheads
            assert self.config.nheads % self.config.kvheads == 0

        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=self.config.use_bias,
        )

        self.ff_sub_layer = FeedForwardBlock(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=self.config.use_bias,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x,
        *,
        mask=None,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        is_causal_mask=False,
        attn_algorithm=None,
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

        self.embedding = AbsolutePositionEmbedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            self.config.pad_id,
            self.config.max_pos,
        )
        self.dec_norm = nn.LayerNorm(self.config.emb_dim, eps=self.config.ln_eps)

        if self.config.emb_dropout:
            self.emb_dropout = nn.Dropout(self.config.emb_dropout)

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x: Optional[torch.LongTensor] = None,
        mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[
            Tuple[
                torch.FloatTensor,
            ]
        ] = None,
        use_cache: bool = True,
        attn_algorithm: Optional[str] = None,
    ):
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len

        if x is not None and inputs_embeds is not None:
            raise ValueError(
                f"You cannot specify both x_in and inputs_embeds at the same time"
            )
        elif x is not None:
            qlen = x.size(1)
            klen = x.size(1)
        elif inputs_embeds is not None:
            qlen = inputs_embeds.size(1)
            klen = inputs_embeds.size(1)
        elif x is None and inputs_embeds is None:
            raise ValueError(f"You have to specify either x_in or inputs_embeds")

        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        # if we are using the cache, the key length needs to be extended with the past keys length
        if use_cache and past_key_value_states[0] is not None:
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
                is_pad: torch.BoolTensor = x == pad_id
                mask: torch.BoolTensor = is_pad.unsqueeze(-1) == is_pad.unsqueeze(-2)
                mask = mask.tril(diagonal=0)

        if x is not None:
            # if position ids is none, we assume this will be a range from 0:qlen then add the past klen => klen:qlen+klen
            # when we have left padding in batches, position_ids must be provided in order to account for the the number of pads in each sequence in the batch
            # position_ids provided do not require any correction
            if position_ids is None:
                # Compute position_ids based on cache config
                _position_ids = torch.arange(
                    0, qlen, dtype=torch.long, device=x.device
                ).repeat(x.size(0), 1)
                if use_cache and past_key_value_states[0] is not None:
                    _position_ids += past_key_value_states[0][0].shape[2]
            else:
                _position_ids = position_ids
            x = self.embedding(
                x, position_ids=_position_ids, correct_pads=position_ids is None
            )
        else:
            x = inputs_embeds

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
        self.config: GPTBigCodeConfig = self.config.updated(**kwargs)

        self.base_model = GPTBigCodeHeadless(self.config)
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=self.config.vocab_bias
        )

        # this model ties weights, so we tie here
        self.head.weight = self.base_model.embedding.emb.weight

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
        x=None,
        mask=None,
        inputs_embeds=None,
        past_key_value_states=None,
        use_cache=False,
        attn_algorithm: Optional[str] = None,
    ):
        output, cache = self.base_model(
            x,
            mask,
            inputs_embeds,
            past_key_value_states=past_key_value_states,
            use_cache=use_cache,
            attn_algorithm=attn_algorithm,
        )

        preds = self.head(output)

        if use_cache:
            return preds, cache
        else:
            return preds
