from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from fms.modules.positions import PositionEncoder


class MultiHeadAttention(nn.Module):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    ...
    Args
    ----
    emb_dim : int
        Latent dimensionality of input and output tensors.
    emb_kq : int
        Latent dimensionality of each head in key and query projections (attention dimension).
    emb_v : int
        Latent dimensionality of each head in value projection (mixing dimension).
    nheads : int
        Number of attention heads.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    factorable_emb: Optional[Callable]
        Function that computes factorable embeddings (like RoPE). It is mutually exclusive with
        additive biases on forward() passed as rel_pos_bias
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        gain=1,
    ):
        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.query = nn.Linear(
            self.emb_dim, self.nheads * self.emb_kq_per_head, bias=use_bias
        )
        self.key = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_kq_per_head, bias=use_bias
        )
        self.value = nn.Linear(
            self.emb_dim, self.kvheads * self.emb_v_per_head, bias=use_bias
        )
        self.dense = nn.Linear(
            self.nheads * self.emb_v_per_head, self.emb_dim, bias=use_bias
        )
        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        self.position_encoder = position_encoder
        # Avoiding graph breaks
        self.previous_flash: bool = torch.backends.cuda.flash_sdp_enabled()
        self.previous_mem_efficient: bool = (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        )
        self.previous_math: bool = torch.backends.cuda.math_sdp_enabled()
        self.reset_params(gain)

    def reset_params(self, gain=1):
        # Ensure softmax inputs are standard normal
        for layer in ["query", "key"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight, mean=0.0, std=self.emb_dim**-0.5
            )
        # Ensure projection layers have same scale (for normalized-step dataloaders like
        # AdamW / Sophia), and maintain input norm up to attention remix, in expectation
        for layer in ["value", "dense"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=(gain / (self.emb_dim * self.nheads * self.emb_v_per_head) ** 0.5)
                ** 0.5,
            )  # Using explicit terms instead of numel to account for eventual MQA addition
        if self.use_bias:
            for layer in ["query", "key", "value", "dense"]:
                getattr(self, layer).bias.data.zero_()

    def forward(
        self,
        q,
        k,
        v,
        mask=None,
        attn_algorithm=None,
        past_key_value_state=None,
        use_cache=False,
        is_self=True,
        is_causal_mask=False,
    ):
        """
        past_key_value_state: tuple
            the cache to be used in attention of the form (<self/cross>_key, <self/cross>_value)
        use_cache: bool
            if True, the kv states for self/cross attention will be saved, otherwise they will not be saved
        is_self: bool
            if True, this will perform self attention, otherwise this will perform cross attention. Note: This will
            only be used in the case that use_cache=True. This may be removed in future

        Returns
        -------
        tensor or tuple
            If use_cache=False, only the hidden state will be returned as a tensor. If use_cache=True, a tuple will be
            returned in the form (hidden_state, cache) where hidden_state is a tensor and cache is of the form specified
            in past_key_value_state
        """

        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()
        kv_len = k.size(1)

        # split emb_dim as nheads*emb_dim_per_head
        # b x h x qlen x ds
        queries = self.query(q).view(
            batch_size, q_len, self.nheads, self.emb_kq_per_head
        )
        queries = queries.transpose(2, 1)  # / (self.emb_kq_per_head**(1/4))

        # if this is self attention, we always recompute
        # cross attention only gets computed when a cache does not exist
        # if we dont have the cache yet, we need to compute
        # d x (h x ds)
        # b x kvlen x d
        # b x kvlen x h x ds
        # b x h x kvlen x ds
        if is_self or past_key_value_state is None:
            keys = self.key(k).view(
                batch_size, kv_len, self.kvheads, self.emb_kq_per_head
            )
            keys = keys.transpose(2, 1)  # / (self.emb_kq_per_head**(1/4))

            values = self.value(v).view(
                batch_size, kv_len, self.kvheads, self.emb_v_per_head
            )
            values = values.transpose(2, 1)  # compatible with QK.T

            # You want to apply rotary embeddings pre-cache
            if self.position_encoder is not None:
                queries, keys = self.position_encoder.adjusted_qk(
                    queries, keys, past_key_value_state, use_cache
                )

        # if you want to use caching and past_key_value_state is not None meaning you have values in your cache
        if use_cache and past_key_value_state is not None:
            if is_self:
                keys = torch.cat((past_key_value_state[0], keys), dim=2)
                values = torch.cat((past_key_value_state[1], values), dim=2)
            else:
                keys = past_key_value_state[0]
                values = past_key_value_state[1]

        # Merge rel pos bias and mask into single float mask
        if mask is not None:
            # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
            # we need to create the nheads dimension
            while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
                mask = mask.unsqueeze(1)

        if self.position_encoder is not None:
            attn_mask = self.position_encoder.adjusted_mask(
                mask, queries, keys, past_key_value_state, use_cache
            )
        else:
            attn_mask = mask

        # Expand kv so black-box attn will work
        expansion = self.nheads // self.kvheads
        # k/v: b h l d
        if expansion != 1:
            keys = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
            values = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)

        if attn_algorithm:
            # Pick which fused attn kernels will run.
            use_flash = attn_algorithm == "flash"
            use_mem_efficient = attn_algorithm == "mem"
            use_math = attn_algorithm == "math"

            torch.backends.cuda.enable_flash_sdp(use_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(use_mem_efficient)
            torch.backends.cuda.enable_math_sdp(use_math)

        attn = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            dropout_p=self.p_dropout if self.training else 0.0,
            is_causal=is_causal_mask,
        )

        if attn_algorithm:
            torch.backends.cuda.enable_flash_sdp(self.previous_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(self.previous_mem_efficient)
            torch.backends.cuda.enable_math_sdp(self.previous_math)

        # attn: bs x seq_len x nheads*emb_v_per_head
        # attn: b x h x qlen x ds
        # attn after permute: b x qlen x h x ds
        # b x qlen x (d)
        attn = (
            attn.transpose(2, 1)
            .contiguous()
            .view(batch_size, q_len, self.nheads * self.emb_v_per_head)
        )
        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            return out, (keys, values)
        else:
            return out
