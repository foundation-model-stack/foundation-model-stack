import abc
from typing import Mapping, Optional, Set, Tuple, Any

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
from fms.modules.linear import get_linear


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
        *args,
        **kwargs,
    ):
        linear_config = kwargs.pop("linear_config") if "linear_config" in kwargs else None

        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
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
        *args,
        **kwargs,
    ):
        linear_config = kwargs.pop("linear_config") if "linear_config" in kwargs else None

        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
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
        result = UnfusedQKV(
            self.emb_dim,
            self.nheads,
            self.kvheads,
            self.emb_kq_per_head,
            self.emb_v_per_head,
            self.use_bias,
        ).to(self.qkv_fused.weight.device)
        query, key, value = torch.split(self.qkv_fused.weight, self.splits, dim=0)
        result.query.weight.copy_(query)
        result.key.weight.copy_(key)
        result.value.weight.copy_(value)
        if self.use_bias:
            query_bias, key_bias, value_bias = torch.split(
                self.qkv_fused.bias, self.splits, dim=0
            )
            result.query.bias.copy_(query_bias)
            result.key.bias.copy_(key_bias)
            result.value.bias.copy_(value_bias)
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
        if True, qkv weights will be fused, otherwise qkv weights will be unfused
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
        linear_config: Mapping[str, Any] | None = None,
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

        # if this is self attention, we always recompute
        # cross attention only gets computed when a cache does not exist
        # if we dont have the cache yet, we need to compute
        # d x (h x ds)
        # b x kvlen x d
        # b x kvlen x h x ds
        # b x h x kvlen x ds
        # todo: Cross attention (This always is true for now)
        if is_self or past_key_value_state is None:
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

        queries = queries.transpose(2, 1)  # / (self.emb_kq_per_head**(1/4))
        keys = keys.transpose(2, 1)  # / (self.emb_kq_per_head**(1/4))
        values = values.transpose(2, 1)  # compatible with QK.T

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
        # k/v: b h l d
        if expansion != 1:
            keys_e = keys.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
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
        linear_config: Mapping[str, Any] | None = None,
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
            fused,
            linear_config,
        )
        self.pre_tp_nheads = nheads
        self.pre_tp_kvheads = kvheads
        self.setup_tp(rank, world_size)

    def load_weights(
        self,
        tensor_values: dict[str, torch.Tensor],
    ):
        used_keys: Set[str] = set()
        dense_weight = self._get_sd_weight(
            tensor_values, used_keys, ["dense", "weight"]
        )
        if self.use_bias:
            dense_bias = self._get_sd_weight(
                tensor_values, used_keys, ["dense", "bias"]
            )

        # 1. Grab the weights from tensor_values
        if self.fused:
            qkv_weight = self._get_sd_weight(
                tensor_values, used_keys, ["qkv_fused", "weight"]
            )
            if self.use_bias:
                qkv_bias = self._get_sd_weight(
                    tensor_values, used_keys, ["qkv_fused", "bias"]
                )

            # 2. Raise exceptions
            if len(tensor_values) > (4 if self.use_bias else 2):
                unused_keys = set(tensor_values.keys()).difference(used_keys)
                raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

            # 3. Load and shard the weights
            # The number in max_partition_sizes will signify the largest world size
            # til we need to duplicate.  For instance if we have nheads=16 and
            # world_size=32, then first 2 ranks will get first 1/16th of query
            self.sharded_copy(
                self.in_proj.qkv_fused.weight,
                qkv_weight,
                0,
                [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
            )
            self.sharded_copy(self.dense.weight, dense_weight, 1, [self.world_size])
            if self.use_bias:
                self.sharded_copy(
                    self.in_proj.qkv_fused.bias,
                    qkv_bias,
                    0,
                    [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
                )
                self.sharded_copy(
                    self.dense.bias, dense_bias, 1, [self.world_size], False
                )

        else:
            query_weight = self._get_sd_weight(
                tensor_values, used_keys, ["query", "weight"]
            )
            key_weight = self._get_sd_weight(
                tensor_values, used_keys, ["key", "weight"]
            )
            value_weight = self._get_sd_weight(
                tensor_values, used_keys, ["value", "weight"]
            )

            if self.use_bias:
                query_bias = self._get_sd_weight(
                    tensor_values, used_keys, ["query", "bias"]
                )
                key_bias = self._get_sd_weight(
                    tensor_values, used_keys, ["key", "bias"]
                )
                value_bias = self._get_sd_weight(
                    tensor_values, used_keys, ["value", "bias"]
                )

            # 2. Raise exceptions
            if len(tensor_values) > (8 if self.use_bias else 4):
                unused_keys = set(tensor_values.keys()).difference(used_keys)
                raise AttributeError(f"Unused weight(s): {', '.join(unused_keys)}")

            # 3. Load and shard the weights
            # The number in max_partition_sizes will signify the largest world size
            # til we need to duplicate.  For instance if we have nheads=16 and
            # world_size=32, then first 2 ranks will get first 1/16th of query
            self.sharded_copy(
                self.in_proj.query.weight, query_weight, 0, [self.pre_tp_nheads]
            )
            self.sharded_copy(
                self.in_proj.key.weight, key_weight, 0, [self.pre_tp_kvheads]
            )
            self.sharded_copy(
                self.in_proj.value.weight, value_weight, 0, [self.pre_tp_kvheads]
            )
            self.sharded_copy(self.dense.weight, dense_weight, 1, [self.world_size])
            if self.use_bias:
                self.sharded_copy(
                    self.in_proj.query.bias, query_bias, 0, [self.pre_tp_nheads]
                )
                self.sharded_copy(
                    self.in_proj.key.bias, key_bias, 0, [self.pre_tp_kvheads]
                )
                self.sharded_copy(
                    self.in_proj.value.bias, value_bias, 0, [self.pre_tp_kvheads]
                )
                self.sharded_copy(
                    self.dense.bias, dense_bias, 1, [self.world_size], False
                )

    def load_qparams(
        self,
        tensor_values: dict[str, torch.Tensor],
    ):
        """
        Copy sharded quantization parameters onto sharded linear modules

                          |     GPU     |
        module | qparam   | shard | dim |
        -------+----------+-------+-----|
        QKV    | qweight  |   Y   |  1  |
               | bias     |   Y   |  0  |
               | scales   |   Y   |  1  |
               | qzeros   |   Y   |  1  |
               | g_idx    |   N   |  -  |
        -------+----------+-------+-----|
        dense  | qweight  |   Y   |  0  |
               | bias     |   N   |  -  |
               | scales   |   Y   |  0  |
               | qzeros   |   Y   |  0  |
               | g_idx    |   Y   |  0  |
        """

        if self.fused:
            modules = ["qkv_fused", "dense"]
        else:
            modules = ["query", "value", "key", "dense"]
        qparams = ["qweight", "scales", "qzeros", "g_idx"]
        if self.use_bias:
            qparams.append("bias")
        all_params = {}
        used_keys: Set[str] = set()

        # Collect quantization parameters to copy on sharded module
        tensor_device = None
        for module in modules:
            for qparam in qparams:
                # TODO: reusing method '_get_sd_weight' but consider changing function name
                if module not in all_params:
                    all_params[module] = {}
                all_params[module][qparam] = self._get_sd_weight(
                    tensor_values, used_keys, [module, qparam]
                )
                if tensor_device is None:
                    tensor_device = all_params[module][qparam].device

        # Define sharding parameters
        if self.fused:
            name_to_module = {
                "qkv_fused": self.in_proj.qkv_fused,
                "dense": self.dense,
            }
            shard_dim = {
                "qkv_fused": 1,
                "dense": 0,
            }
            max_partition = {
                "qkv_fused": [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
                "dense": [self.world_size],
            }
        else:
            name_to_module = {
                "query": self.in_proj.query,
                "value": self.in_proj.value,
                "key": self.in_proj.key,
                "dense": self.dense,
                }
            shard_dim = {
                "query": 1,
                "value": 1,
                "key": 1,
                "dense": 0,
            }
            max_partition = {
                "query": [self.pre_tp_nheads],
                "value": [self.pre_tp_kvheads],
                "key": [self.pre_tp_kvheads],
                "dense": [self.world_size],
            }

        # Shard module, one parameter at the time
        for module in modules:
            for qparam in qparams:
                module_qparam = getattr(name_to_module[module], qparam)
                if module_qparam.device == torch.device("meta"):  # TODO: improve cast (no Parameters here, only buffers)
                    setattr(name_to_module[module], qparam, torch.empty_like(module_qparam, device=tensor_device))
                    module_qparam = getattr(name_to_module[module], qparam)
                is_sharded = (
                    False
                    if (
                        (qparam == "bias" and module == "dense") or
                        (qparam == "g_idx" and module != "dense")
                    )
                    else True
                )

                shard_dim_qparam = shard_dim[module]
                if qparam in ["bias", "g_idx"]:
                    shard_dim_qparam = 0

                self.sharded_copy(
                    param=module_qparam,
                    tensor_value=all_params[module][qparam],
                    dim=shard_dim_qparam,
                    max_partition_sizes=max_partition[module],
                    is_sharded=is_sharded,
                )
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
        )
        return tp_mha

    def _copy_to_tp_region(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ):
        if (k is None and v is None) or (k is q and v is q):
            q_par = copy_to_tensor_model_parallel_region(q)
            if self.fused:
                k_par = None
                v_par = None
            else:
                k_par = copy_to_tensor_model_parallel_region(k)
                v_par = copy_to_tensor_model_parallel_region(v)
        else:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        return q_par, k_par, v_par

    def forward(
        self,
        q,
        k=None,
        v=None,
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

        q_par, k_par, v_par = self._copy_to_tp_region(q, k, v)

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
