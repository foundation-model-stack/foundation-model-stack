import abc
from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Tuple, TypedDict

import torch
import torch.distributed
from torch import Tensor, nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.linear import (
    LinearModuleShardingInfo,
    get_all_linear_type_to_sharding_maps,
    get_linear,
    get_linear_type,
)
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule

from torch.library import custom_op

__type_factory_map: dict[str, dict[str, callable]] = {}


# FIXME: add adjusted_mask for alibi as part of attn_compute_dict
def register_attention_op(
    attn_type: str,
    store_op: callable,
    compute_op: callable,
    is_prefill_op: Optional[callable] = None,
    compute_decode_op: Optional[callable] = None,
) -> None:
    if attn_type in __type_factory_map:
        raise KeyError(
            f"Module mapping of attention type `{attn_type}` already registered"
        )
    if compute_decode_op is None:
        compute_decode_op = compute_op

    compute_dict = {
        "store": store_op,
        "is_prefill": (lambda _: True) if is_prefill_op is None else is_prefill_op,
        "compute_prefill": compute_op,
        "compute_decoder": compute_decode_op,
    }
    __type_factory_map[attn_type] = compute_dict


@dataclass
class AttentionKwargs:
    def update(self) -> "AttentionKwargs":
        pass


@dataclass
class SDPAAttentionKwargs(AttentionKwargs):
    attn_name: str = "sdpa"
    mask: Optional[torch.tensor] = None
    attn_algorithm: Optional[str] = None
    is_causal_mask: bool = False

    def update(self) -> "SDPAAttentionKwargs":
        if self.mask is not None:
            mask = self.mask
            # get the last row of the 3d mask
            mask = mask[:, -1:, :]
            # extend the mask one slot
            mask = torch.cat(
                (
                    mask,
                    torch.zeros(mask.size(0), 1, 1, device=mask.device),
                ),
                dim=2,
            )
            if torch._dynamo.config.dynamic_shapes:
                torch._dynamo.mark_dynamic(mask, 2)
        else:
            mask = None
        return SDPAAttentionKwargs(
            mask=mask, attn_algorithm=self.attn_algorithm, is_causal_mask=False
        )


def _sdpa_store_op(
    keys: torch.Tensor,
    values: torch.Tensor,
    key_cache: Optional[torch.Tensor],
    value_cache: Optional[torch.Tensor],
    **_,
):
    # This should probably be in sdpa compute, but more efficient here
    keys = keys.transpose(2, 1)  # / (self.emb_kq_per_head**(1/4))
    values = values.transpose(2, 1)  # compatible with QK.T

    if key_cache is not None and value_cache.numel() > 0:
        return (
            torch.cat((key_cache, keys), dim=2),
            torch.cat((value_cache, values), dim=2),
        )
    else:
        return (keys, values)


def _sdpa_compute_op(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: float,
    attn_kwargs: SDPAAttentionKwargs,
    **_,
):
    queries = query.transpose(2, 1)
    mask = attn_kwargs.mask  # / (self.emb_kq_per_head**(1/4))

    # Merge rel pos bias and mask into single float mask
    if mask is not None:
        # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
        # we need to create the nheads dimension
        while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
            mask = mask.unsqueeze(1)

    # Expand kv so black-box attn will work
    expansion = nheads // kvheads
    # k/v: b h l d
    if expansion != 1:
        keys_e = key_cache.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        values_e = (
            value_cache.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        )
    else:
        keys_e = key_cache
        values_e = value_cache

    # sdpa_kernels = []
    # if attn_kwargs.attn_algorithm is None or attn_kwargs.attn_algorithm == "math":
    #     sdpa_kernels.append(SDPBackend.MATH)
    # elif attn_kwargs.attn_algorithm == "flash":
    #     sdpa_kernels.append(SDPBackend.FLASH_ATTENTION)
    # elif attn_kwargs.attn_algorithm == "mem":
    #     sdpa_kernels.append(SDPBackend.EFFICIENT_ATTENTION)
    # else:
    #     raise ValueError(f"invalid attn_algorithm given: {attn_kwargs.attn_algorithm}")
    
    attn_mask = mask
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(dtype=queries.dtype)
    
    # with sdpa_kernel(sdpa_kernels):

    attn = F.scaled_dot_product_attention(
        queries,
        keys_e,
        values_e,
        attn_mask=attn_mask,
        dropout_p=p_dropout,
        is_causal=attn_kwargs.is_causal_mask,
        scale=scale_factor,
    )

    # attn: bs x seq_len x nheads*emb_v_per_head
    # attn: b x h x qlen x ds
    # attn after permute: b x qlen x h x ds
    # b x qlen x (d)
    attn = attn.transpose(2, 1).contiguous()
    return attn


register_attention_op("sdpa", _sdpa_store_op, _sdpa_compute_op)


def get_attention_type(attn_kwargs: AttentionKwargs) -> dict[str, callable]:
    attn_type = attn_kwargs.attn_name
    if attn_kwargs.attn_name not in __type_factory_map:
        # we can add sdpa default here
        raise KeyError("")

    return __type_factory_map[attn_type]


class QKV(nn.Module, metaclass=abc.ABCMeta):
    """Simple module for applying qkv in attention"""

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_kq_per_head = emb_kq_per_head
        self.emb_v_per_head = emb_v_per_head
        self.use_bias = use_bias
        self.linear_config = linear_config

    @abc.abstractmethod
    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """applies query/key/value transformations on q, k, v inputs respectively and returns the resulting values

        Args:
            q: torch.Tensor
                the query tensor
            k: Optional[torch.Tensor]
                the optional key tensor
            v: Optional[torch.Tensor]
                the optional value tensor

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            the query, key, and value computed
        """
        pass

    @abc.abstractmethod
    def reset_parameters(self):
        """resets the query, key, and value weights for training

        Args:
            gain: int
                gain for std in norm (default is 1)
        """
        pass


class UnfusedQKV(QKV):
    """
    Unfused Weights implementation of QKV
    """

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
            linear_config,
            *args,
            **kwargs,
        )

        self.query = get_linear(
            self.emb_dim,
            self.nheads * self.emb_kq_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.key = get_linear(
            self.emb_dim,
            self.kvheads * self.emb_kq_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.value = get_linear(
            self.emb_dim,
            self.kvheads * self.emb_v_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()

    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if k is None and v is None:
            k = q
            v = q
        elif k is None or v is None:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        # b x h x qlen x ds
        queries = self.query(q)
        keys = self.key(k)
        values = self.value(v)
        return queries, keys, values


class FusedQKV(QKV):
    """
    Fused Weights implementation of QKV
    """

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
            linear_config,
            *args,
            **kwargs,
        )
        self.splits = [
            self.nheads * self.emb_kq_per_head,
            self.kvheads * self.emb_kq_per_head,
            self.kvheads * self.emb_v_per_head,
        ]

        self.qkv_fused = get_linear(
            self.emb_dim,
            sum(self.splits),
            bias=self.use_bias,
            linear_config=linear_config,
        )

    def unfuse_weights(self):
        with torch.device("meta"):
            result = UnfusedQKV(
                self.emb_dim,
                self.nheads,
                self.kvheads,
                self.emb_kq_per_head,
                self.emb_v_per_head,
                self.use_bias,
            )
        query, key, value = torch.split(self.qkv_fused.weight, self.splits, dim=0)
        result.query.weight = torch.nn.Parameter(query)
        result.key.weight = torch.nn.Parameter(key)
        result.value.weight = torch.nn.Parameter(value)
        if self.use_bias:
            query_bias, key_bias, value_bias = torch.split(
                self.qkv_fused.bias, self.splits, dim=0
            )
            result.query.bias = torch.nn.Parameter(query_bias)
            result.key.bias = torch.nn.Parameter(key_bias)
            result.value.bias = torch.nn.Parameter(value_bias)
        return result

    def reset_parameters(self):
        nn.init.trunc_normal_(self.qkv_fused.weight, mean=0.0, std=0.02)
        if self.use_bias:
            self.qkv_fused.bias.data.zero_()

    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (k is None and v is None) or (k is q and v is q):
            qkv = q
        else:
            raise ValueError("q, k, and v must be the same or k and v must be None")
        return self.qkv_fused(qkv).split(self.splits, dim=-1)


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
    fused : bool
        If True, qkv weights will be fused, otherwise qkv weights will be unfused.
    linear_config : Mapping[str, Any] | None
        Configuration for selection of linear modules (QKV, dense).
        Pass as {"linear_type": [str | callable], <other kwargs>}.
        "linear_type" should provide the string identifier of a registered type
        (e.g., "torch_linear", "gptq", ...) or a callable for module selection depending
        on module name. Additional config options should be provided as kwargs in
        linear_config.
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
        fused: bool = True,
        linear_config: Optional[Mapping[str, Any]] = None,
        scale_factor: Optional[float] = None,
    ):
        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.fused = fused
        self.linear_config = linear_config
        self.scale_factor = scale_factor

        self.in_proj: QKV = (FusedQKV if self.fused else UnfusedQKV)(
            self.emb_dim,
            self.nheads,
            self.kvheads,
            self.emb_kq_per_head,
            self.emb_v_per_head,
            self.use_bias,
            linear_config=linear_config,
        )

        self.dense = get_linear(
            self.nheads * self.emb_v_per_head,
            self.emb_dim,
            bias=use_bias,
            linear_config=linear_config,
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

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, QKV):
                m.reset_parameters()

    def to_tp(self, group: ProcessGroup) -> "TPMultiHeadAttention":
        return TPMultiHeadAttention.import_module(self, group)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache=False,
        attn_kwargs: Optional[AttentionKwargs] = None,
    ):
        """
        past_key_value_state: tuple
            the cache to be used in attention of the form (<self/cross>_key, <self/cross>_value)
        position_ids: Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. Used for RoPE embeddings
        use_cache: bool
            if True, the kv states for self/cross attention will be saved, otherwise they will not be saved

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

        # if this is self attention, we always recompute
        # cross attention only gets computed when a cache does not exist
        # if we dont have the cache yet, we need to compute
        # d x (h x ds)
        # b x kvlen x d
        # b x kvlen x h x ds
        # b x h x kvlen x ds
        # todo: Cross attention (This always is true for now)
        q_out, k_out, v_out = self.in_proj(q, k, v)

        # note: transposes will be moved in a later PR to fix dis-contiguous tensor issues
        queries = q_out.view(batch_size, q_len, self.nheads, self.emb_kq_per_head)
        keys = k_out.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head)
        values = v_out.view(batch_size, q_len, self.kvheads, self.emb_v_per_head)

        # You want to apply rotary embeddings pre-cache
        if self.position_encoder is not None:
            queries, keys = self.position_encoder.adjusted_qk(
                queries, keys, position_ids, past_key_value_state, use_cache
            )

        attn_compute_dict = get_attention_type(attn_kwargs)

        # this can occur during prefill or use_cache=False
        if past_key_value_state is None:
            past_key_value_state = (None, None)

        keys, values = attn_compute_dict["store"](
            keys,
            values,
            past_key_value_state[0],
            past_key_value_state[1],
            attn_kwargs=attn_kwargs,
        )

        if attn_compute_dict["is_prefill"](attn_kwargs):
            attn = attn_compute_dict["compute_prefill"](
                queries,
                keys,
                values,
                self.nheads,
                self.kvheads,
                self.p_dropout,
                self.scale_factor,
                attn_kwargs=attn_kwargs,
            )
        else:
            attn = attn_compute_dict["compute_decode"](
                queries,
                keys,
                values,
                self.nheads,
                self.kvheads,
                self.p_dropout,
                self.scale_factor,
                attn_kwargs=attn_kwargs,
            )

        attn = attn.view(batch_size, q_len, self.nheads * self.emb_v_per_head)
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
        fused: bool = True,
        group: Optional[ProcessGroup] = None,
        linear_config: Optional[Mapping[str, Any]] = None,
        scale_factor: Optional[float] = None,
    ):
        assert torch.distributed.is_initialized()

        rank, world_size = distributed.rank_and_world(group)
        assert nheads % world_size == 0, (
            "The number of heads must be divisible by world size"
        )
        assert (kvheads >= world_size and kvheads % world_size == 0) or (
            kvheads < world_size and world_size % kvheads == 0
        ), (
            "the kv heads must be divisible by the world size or the world size must be divisible by kv heads"
        )
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
            fused,
            linear_config,
            scale_factor,
        )
        self.pre_tp_nheads = nheads
        self.pre_tp_kvheads = kvheads
        self.setup_tp(rank, group)

        # linear_type must handle module_name = None to support TP of MHA
        self.linear_type = get_linear_type(self.linear_config)

    def load_weights(
        self,
        tensor_values: dict[str, torch.Tensor],
    ) -> Optional[set]:
        """Define sharding info of MHA module as:
        {'module_name': (module_obj, sharding_dim, max_partition)}
        Then, call the pre-registered sharding function associated with
        self.linear_type.

        `sharding_dim` is sharding dimension of the `weights` parameter
        of nn.Linear. It may differ for other types of linear or other
        parameters.

        The numbers in `max_partition` signify the largest world size
        till we need to duplicate. For instance if we have nheads=16 and
        world_size=32, then first 2 ranks will get first 1/16th of query
        """

        if self.fused:
            module_sharding_info = {
                "qkv_fused": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("qkv_fused"),
                    0,
                    [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
                ),
                "dense": LinearModuleShardingInfo(self.dense, 1, [self.world_size]),
            }
        else:
            module_sharding_info = {
                "query": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("query"), 0, [self.pre_tp_nheads]
                ),
                "key": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("key"), 0, [self.pre_tp_kvheads]
                ),
                "value": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("value"), 0, [self.pre_tp_kvheads]
                ),
                "dense": LinearModuleShardingInfo(self.dense, 1, [self.world_size]),
            }

        type_sharding_map = get_all_linear_type_to_sharding_maps()
        unused_keys = type_sharding_map[self.linear_type](
            tensor_values,
            self,
            module_sharding_info,
        )
        return unused_keys

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
            fused=mha.fused,
            linear_config=mha.linear_config,
            scale_factor=mha.scale_factor,
        )
        return tp_mha

    def _copy_to_tp_region(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ):
        if (k is None and v is None) or (k is q and v is q):
            q_par = copy_to_tensor_model_parallel_region(q, self.group)
            if self.fused:
                k_par = None
                v_par = None
            else:
                k_par = copy_to_tensor_model_parallel_region(k, self.group)
                v_par = copy_to_tensor_model_parallel_region(v, self.group)
        else:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        return q_par, k_par, v_par

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_value_state: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache=False,
        attn_kwargs: Optional[AttentionKwargs] = None,
    ):
        """
        Check MultiHeadAttention for up-to-date arguments and docs
        """

        q_par, k_par, v_par = self._copy_to_tp_region(q, k, v)

        out_par = MultiHeadAttention.forward(
            self,
            q_par,
            k_par,
            v_par,
            position_ids,
            past_key_value_state,
            use_cache,
            attn_kwargs,
        )

        # if use_cache=True, we return the hidden_state as well as the kv cache.
        # We only reduce the output, and keep the cache thread-local
        if use_cache:
            out = reduce_from_tensor_model_parallel_region(out_par[0], self.group)
            return out, out_par[1]
        else:
            out = reduce_from_tensor_model_parallel_region(out_par, self.group)
            return out
