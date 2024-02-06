import math
from typing import List, Optional, Tuple

import torch
import torch.distributed
from torch import Tensor, nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional as F

from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule
from fms.utils.cache import AttentionComputationMixin, CacheDataLayer


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

        self.splits = [
            self.nheads * self.emb_kq_per_head,
            self.kvheads * self.emb_kq_per_head,
            self.kvheads * self.emb_v_per_head,
        ]

        self.qkv_fused = nn.Linear(
            self.emb_dim,
            sum(self.splits),
            bias=use_bias,
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
        self.head_size = emb_dim // nheads
        self.reset_params(gain)

    def reset_params(self, gain=1):
        # Ensure softmax inputs are standard normal
        self.qkv_fused.weight.data[
        : self.emb_kq_per_head * (self.nheads + self.kvheads)
        ].normal_(0, self.emb_dim ** -0.5)
        # Ensure projection layers have same scale (for normalized-step dataloaders like
        # AdamW / Sophia), and maintain input norm up to attention remix, in expectation
        self.qkv_fused.weight.data[
        self.emb_kq_per_head * (self.nheads + self.kvheads):
        ].normal_(
            0, (gain / (self.emb_dim * self.nheads * self.emb_v_per_head) ** 0.5) ** 0.5
        )
        if self.use_bias:
            for layer in ["qkv_fused", "dense"]:
                getattr(self, layer).bias.data.zero_()

    def forward(
        self,
        qkv,
        mask: Optional[Tensor] = None,
        position_ids=None,
        attn_algorithm=None,
        cache_data_layer=None,
        use_cache=False,
        is_self=True,
        is_causal_mask=False,
    ):
        """
        cache_data_layer: CacheDataLayer, optional
            A single layer of the cache (default is None)
        position_ids: Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. Used for RoPE embeddings
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
        batch_size, q_len, _ = qkv.size()

        if is_self or past_key_value_state is None:
            q, k, v = self.qkv_fused(qkv).split(self.splits, dim=-1)

            queries = q.view(
                batch_size, q_len, self.nheads, self.emb_kq_per_head
            )#.transpose(2, 1)
            keys = k.view(
                batch_size, q_len, self.kvheads, self.emb_kq_per_head
            )#.transpose(2, 1)
            values = v.view(
                batch_size, q_len, self.kvheads, self.emb_v_per_head
            )

            # You want to apply rotary embeddings pre-cache
            if self.position_encoder is not None:
                queries, keys = self.position_encoder.adjusted_qk(
                    queries,
                    keys,
                    position_ids,
                    use_cache,
                )

        # store the values in kv-cache
        if use_cache and cache_data_layer:
            keys, values = cache_data_layer.store(keys, values)

        custom_attention = (
            use_cache
            and cache_data_layer
            and isinstance(cache_data_layer, AttentionComputationMixin)
        )

        # Provide a method for a user to perform their own implementation of attention in the cache case if required
        if custom_attention and cache_data_layer.is_filled():
            attn = cache_data_layer.attend(queries, keys, values)
        # otherwise we always fall back into SDPA as this is either a prompt or it is a single contiguous cache
        else:
            queries = queries.transpose(2, 1)
            keys = keys.transpose(2, 1)
            values = values.transpose(2, 1)

            # Merge rel pos bias and mask into single float mask
            if mask is not None:
                # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
                # we need to create the nheads dimension
                while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
                    mask = mask.unsqueeze(1)

            if self.position_encoder is not None:
                attn_mask = self.position_encoder.adjusted_mask(
                    mask, queries, keys, position_ids, use_cache
                )
            else:
                attn_mask = mask

            # Expand kv so black-box attn will work
            expansion = self.nheads // self.kvheads
            # k/v: b h l d
            if expansion != 1:
                keys_e = (
                    keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
                )
                values_e = (
                    values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
                )
            else:
                keys_e = keys
                values_e = values

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
                keys_e,
                values_e,
                attn_mask=attn_mask,
                dropout_p=self.p_dropout if self.training else 0.0,
                is_causal=is_causal_mask,
            )

            if attn_algorithm:
                torch.backends.cuda.enable_flash_sdp(self.previous_flash)
                torch.backends.cuda.enable_mem_efficient_sdp(
                    self.previous_mem_efficient
                )
                torch.backends.cuda.enable_math_sdp(self.previous_math)

            # attn: bs x seq_len x nheads*emb_v_per_head
            # attn: b x h x qlen x ds
            # attn after permute: b x qlen x h x ds
            # b x qlen x (d)
            attn = attn.transpose(2, 1).contiguous()

        attn = attn.view(batch_size, q_len, self.nheads * self.emb_v_per_head)

        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            # note: needed to add this check to return the data_layer as it fails compile otherwise
            return out, cache_data_layer.data_layer if custom_attention else (
                keys,
                values,
            )
        else:
            return out


class TPMultiHeadAttention(MultiHeadAttention, TPModule):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    This subclass adds support for Tensor Parallel
    ...
    Args
    ----
    Check MultiHeadAttention for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
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
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()

        rank, world_size = distributed.rank_and_world(group)
        assert (
            nheads % world_size == 0
        ), "The number of heads must be divisible by world size"
        MultiHeadAttention.__init__(
            self,
            emb_dim,
            emb_kq,
            emb_v,
            nheads // world_size,
            (kvheads // world_size) if kvheads > 1 else kvheads,
            p_dropout,
            use_bias,
            position_encoder,
            gain,
        )
        self.setup_tp(rank, world_size)
        self.head_size = self.head_size // world_size

    def colwise_param_names(self) -> List[str]:
        colwise_weights = ["query"]
        if self.kvheads != 1:
            colwise_weights.append("key")
            colwise_weights.append("value")
        return colwise_weights

    def rowwise_param_names(self) -> List[str]:
        return ["dense"]

    @staticmethod
    def import_module(
        mha: MultiHeadAttention, group: ProcessGroup
    ) -> "TPMultiHeadAttention":
        tp_mha = TPMultiHeadAttention(
            emb_dim=mha.emb_dim,
            emb_kq=mha.emb_kq_per_head,
            emb_v=mha.emb_v_per_head,
            nheads=mha.nheads,
            kvheads=mha.kvheads,
            p_dropout=mha.p_dropout,
            use_bias=mha.use_bias,
            position_encoder=mha.position_encoder,
            group=group,
        )
        return tp_mha

    def forward(
        self,
        qkv,
        mask=None,
        position_ids=None,
        attn_algorithm=None,
        cache_data_layer=None,
        use_cache=False,
        is_self=True,
        is_causal_mask=False,
    ):
        """
        Check MultiHeadAttention for up-to-date arguments and docs
        """

        q_par = copy_to_tensor_model_parallel_region(q)
        k_par = copy_to_tensor_model_parallel_region(k)
        v_par = copy_to_tensor_model_parallel_region(v)
        # rel_pos_bias_par = copy_to_tensor_model_parallel_region(rel_pos_bias)

        out_par = MultiHeadAttention.forward(
            self,
            q_par,
            k_par,
            v_par,
            mask,
            position_ids,
            attn_algorithm,
            cache_data_layer,
            use_cache,
            is_self,
            is_causal_mask,
        )

        # if use_cache=True, we return the hidden_state as well as the kv cache.
        # We only reduce the output, and keep the cache thread-local
        if use_cache:
            out = reduce_from_tensor_model_parallel_region(out_par[0])
            return out, out_par[1]
        else:
            out = reduce_from_tensor_model_parallel_region(out_par)
            return out
