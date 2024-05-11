from typing import Dict, List, Optional, Set, Tuple

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
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule


def get_scan_plan(x, fmap, h):
    # x: b n d
    # plan: for each level, which entries to avg from previous level ([l] n' 2)
    # inds: which level and entry to pull from in populating heads (n h 2) -> (n h)
    b, n, d = x.size()
    # Form ruler-tick progression sequence
    levels = sum(
        [
            torch.arange(n, device=x.device)
            .remainder(2**i)
            .sub(2**i - 1)
            .sign()
            .add(1)
            for i in range(n.bit_length())
        ]
    ).roll(1, 0)
    plan = [
        torch.zeros(0, 2, device=x.device, dtype=torch.int)
        for _ in range(len(fmap) + 2)
    ]  # [l] 0 2
    plan[1] = (
        torch.arange(x.size(1) + 1, device=x.device, dtype=torch.int)
        .unsqueeze(1)
        .expand(-1, 2)
    )
    inds = torch.zeros(n, h, 2, device=x.device, dtype=torch.long)  # n h 2
    inds[:, 0, 1] = torch.arange(n, device=inds.device, dtype=inds.dtype) + 1
    inds[:, :, 0] = 1
    ilist = list(range(1, n))
    for i in ilist:
        m = fmap.get(levels[i].item(), h)
        inds[i, 1:m] = inds[i - 1, : m - 1]
        if m < h:
            inds[i, m + 1 :] = inds[i - 1, m + 1 :]
            prev = inds[i - 1, m - 1 : m + 1].flip([0])  # 2 2
            assert prev[0, 0] == min(levels[i], len(fmap) + 1) or prev[0, 1] == 0, (
                levels[i],
                prev[0, 0],
            )
            assert prev[1, 0] == min(levels[i], len(fmap) + 1) or prev[1, 1] == 0, (
                levels[i],
                prev[1, 0],
            )
            level = plan[levels[i] + 1]
            inds[i, m, 0] = levels[i] + 1
            inds[i, m, 1] = level.size(0)
            plan[levels[i] + 1] = torch.cat(
                [plan[levels[i] + 1], prev[:, 1][None]], dim=0
            )
    # Flatten inds (indexing into flattened plan/cache) (n h)
    ls = [p.size(0) for p in plan]
    ls = [0] + ls[:-1]
    offset = torch.tensor(ls, device=inds.device).cumsum(0)
    offset = offset[inds[:, :, 0]]
    inds = offset + inds[:, :, 1]
    return plan + [inds]


def scan(x, plan):
    # x: b n d
    b, n, d = x.size()
    inds = plan[-1]
    plan = plan[:-1]

    # Plan and inds are formed, construct cache via recursive sums
    cache = [torch.empty_like(x[:, :0]) for _ in plan]  # [l] b 0 d
    cache[1] = nn.functional.pad(x, (0, 0, 1, 0))  # b n d
    for i in range(2, len(cache)):
        cache[i] = (
            cache[i - 1]
            .index_select(1, plan[i].view(-1))
            .view(b, -1, 2, d)
            .sum(2)
            .div(2**0.5)
        )

    cache = (
        torch.cat(cache, dim=1).unsqueeze(3).expand(-1, -1, -1, inds.size(-1))
    )  # b n' d h
    out = cache.gather(1, inds[None, :, None].expand(b, -1, d, -1))  # b n d h
    return out


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
        # self.ln_k = LayerNormParameterized(emb_kq, use_high_precision_pow=True)
        # self.ln_v = LayerNormParameterized(emb_v, use_high_precision_pow=True)

        self.inp_len = 0
        self.plan = None

        fmap = {8 - i: 64 - (i) ** 2 for i in range(8)}
        fmap.pop(8)
        fmap.pop(7)
        self.fmap = fmap
        self.cache_size = 64

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, LayerNormParameterized):
                m.reset_parameters()

    def to_tp(self, group: ProcessGroup) -> "TPMultiHeadAttention":
        return TPMultiHeadAttention.import_module(self, group)

    def forward(
        self,
        q,
        k,
        v,
        mask: Optional[Tensor] = None,
        position_ids=None,
        attn_algorithm=None,
        past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache=False,
        is_self=True,
        is_causal_mask=False,
    ):
        """
        past_key_value_state: tuple
            the cache to be used in attention of the form (<self/cross>_key, <self/cross>_value)
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
        batch_size, q_len, _ = q.size()
        kv_len = k.size(1)

        # if kv_len mismatch, build new scan plan
        if kv_len != self.inp_len:
            self.inp_len = kv_len
            with torch.no_grad():
                self.plan = get_scan_plan(k, self.fmap, self.cache_size)

        # split emb_dim as nheads*emb_dim_per_head
        # b x h x qlen x ds
        queries = self.query(q).view(
            batch_size, q_len, self.nheads, self.emb_kq_per_head
        )

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

            values = self.value(v).view(
                batch_size, kv_len, self.kvheads, self.emb_v_per_head
            )

            # You want to apply rotary embeddings pre-cache
            if self.position_encoder is not None:
                queries, keys = self.position_encoder.adjusted_qk(
                    queries, keys, position_ids, past_key_value_state, use_cache
                )

        queries = queries / (self.emb_kq_per_head**0.5)  # b l h d

        # Build telescoping cache
        # k/v: b l h d
        keys = keys.view(batch_size, kv_len, -1)
        keys = scan(keys, self.plan).unflatten(
            2, (self.kvheads, self.emb_kq_per_head)
        )  # b l h d 64
        # keys = self.ln_k(keys.transpose(3, 4))  # b l h 64 d
        values = values.view(batch_size, kv_len, -1)
        values = scan(values, self.plan).unflatten(
            2, (self.kvheads, self.emb_v_per_head)
        )  # b l h d 64
        # values = self.ln_v(values.transpose(3, 4))  # b l h 64 d

        # if you want to use caching and past_key_value_state is not None meaning you have values in your cache
        if (
            use_cache
            and past_key_value_state is not None
            and past_key_value_state[0].numel() > 0
        ):
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
            attn_mask: Optional[Tensor] = self.position_encoder.adjusted_mask(
                mask, queries, keys, past_key_value_state, use_cache
            )
        else:
            attn_mask = mask

        # Expand kv so black-box attn will work
        expansion = self.nheads // self.kvheads
        # k/v: b l h d 64
        # q: b l he d
        queries = queries.unflatten(2, (self.kvheads, expansion))
        #     keys_e = (
        #         keys.unsqueeze(3).expand(-1, -1, -1, expansion, -1, -1).flatten(2, 3)
        #     )
        #     values_e = (
        #         values.unsqueeze(3).expand(-1, -1, -1, expansion, -1, -1).flatten(2, 3)
        #     )
        # else:
        #     keys_e = keys
        #     values_e = values

        attn = queries.matmul(keys)  # b l h e 64
        # attn = torch.einsum("blhed,blhdc->blhec", queries, keys_e)
        attn = attn.softmax(4)
        attn = attn.matmul(values.transpose(3, 4))  # b l h e d
        # attn = torch.einsum("blhec,blhdc->blhed", attn, values_e)

        # attn: bs x seq_len x nheads*emb_v_per_head
        # attn: b x h x qlen x ds
        # attn after permute: b x qlen x h x ds
        # b x qlen x (d)
        attn = attn.reshape(batch_size, q_len, self.nheads * self.emb_v_per_head)
        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            return out, (keys, values)
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
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()

        rank, world_size = distributed.rank_and_world(group)
        assert (
            nheads % world_size == 0
        ), "The number of heads must be divisible by world size"
        assert (kvheads >= world_size and kvheads % world_size == 0) or (
            kvheads < world_size and world_size % kvheads == 0
        ), "the kv heads must be divisible by the world size or the world size must be divisible by kv heads"
        MultiHeadAttention.__init__(
            self,
            emb_dim,
            emb_kq,
            emb_v,
            nheads // world_size,
            (kvheads // world_size) if kvheads >= world_size else 1,
            p_dropout,
            use_bias,
            position_encoder,
        )
        self.pre_tp_nheads = nheads
        self.pre_tp_kvheads = kvheads
        self.setup_tp(rank, world_size)

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        # 1. Grab the weights from tensor_values
        used_keys: Set[str] = set()
        query_weight = self._get_sd_weight(
            tensor_values, used_keys, ["query", "weight"]
        )
        key_weight = self._get_sd_weight(tensor_values, used_keys, ["key", "weight"])
        value_weight = self._get_sd_weight(
            tensor_values, used_keys, ["value", "weight"]
        )
        dense_weight = self._get_sd_weight(
            tensor_values, used_keys, ["dense", "weight"]
        )
        if self.use_bias:
            query_bias = self._get_sd_weight(
                tensor_values, used_keys, ["query", "bias"]
            )
            key_bias = self._get_sd_weight(tensor_values, used_keys, ["key", "bias"])
            value_bias = self._get_sd_weight(
                tensor_values, used_keys, ["value", "bias"]
            )
            dense_bias = self._get_sd_weight(
                tensor_values, used_keys, ["dense", "bias"]
            )

        # 2. Raise exceptions
        if len(tensor_values) > (8 if self.use_bias else 4):
            unused_keys = set(tensor_values.keys()).difference(used_keys)
            raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

        # 3. Load and shard the weights
        # The number in max_partition_sizes will signify the largest world size
        # til we need to duplicate.  For instance if we have nheads=16 and
        # world_size=32, then first 2 ranks will get first 1/16th of query
        self.sharded_copy(self.query.weight, query_weight, 0, [self.pre_tp_nheads])
        self.sharded_copy(self.key.weight, key_weight, 0, [self.pre_tp_kvheads])
        self.sharded_copy(self.value.weight, value_weight, 0, [self.pre_tp_kvheads])
        self.sharded_copy(self.dense.weight, dense_weight, 1, [self.world_size])
        if self.use_bias:
            self.sharded_copy(self.query.bias, query_bias, 0, [self.pre_tp_nheads])
            self.sharded_copy(self.key.bias, key_bias, 0, [self.pre_tp_kvheads])
            self.sharded_copy(self.value.bias, value_bias, 0, [self.pre_tp_kvheads])
            self.sharded_copy(self.dense.bias, dense_bias, 1, [self.world_size], False)

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
        q,
        k,
        v,
        mask=None,
        position_ids=None,
        attn_algorithm=None,
        past_key_value_state=None,
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
            past_key_value_state,
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
