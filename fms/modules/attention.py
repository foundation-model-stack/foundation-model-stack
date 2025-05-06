import abc
import math
from typing import Any, Mapping, Optional, Tuple

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
from fms.modules.linear import (
    LinearModuleShardingInfo,
    get_all_linear_type_to_sharding_maps,
    get_linear,
    get_linear_type,
)
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule

# New PyTorhch Paged-Attention class API (nightly ≥ 20240420) built upon Flex Attention.
# For reference, see:
# https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/experimental/_paged_attention.py
from torch.nn.attention.experimental._paged_attention import PagedAttention
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

_DEFAULT_SPARSE_BLOCK_SIZE = 128


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
    paged_attention_config : dict | None
        Optional settings for paged attention (e.g. {"block_size": 128, "max_blocks": 2048}).
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
        paged_attention_config: Optional[dict] = None,
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
        self.paged_attention_config = paged_attention_config or {}

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

        # ------------------------------------------------------------------
        # Lazy-initialised PagedAttention manager & KV caches
        # ------------------------------------------------------------------
        self._paged_mgr = None          # type: Optional[PagedAttention]
        self._k_cache = None            # type: Optional[torch.Tensor]
        self._v_cache = None            # type: Optional[torch.Tensor]

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

    def _validate_paged_attention(
        self,
        seq_len: int,
        block_size: int,
        max_blocks: Optional[int] = None
    ) -> None:
        """Validates paged attention parameters.
        
        Args:
            seq_len: Sequence length
            block_size: Block size
            max_blocks: Maximum allowed blocks (from config)
        """
        if seq_len <= 0:
            raise ValueError("Sequence length must be positive.")
        if (block_size & (block_size - 1)) != 0:
            raise ValueError(f"Block size {block_size} must be a power of 2")
            
        num_blocks = (seq_len + block_size - 1) // block_size
        if max_blocks and num_blocks > max_blocks:
            raise ValueError(
                f"Sequence length {seq_len} with block size {block_size} "
                f"exceeds maximum allowed blocks {max_blocks}"
            )

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
        block_size: Optional[int] = None,
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
            # When QKV weights are fused we only need the query tensor; the
            # fused linear produces K and V internally.
            if self.fused:
                q_out, k_out, v_out = self.in_proj(q, None, None)
            else:
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
            if is_self and attn_algorithm != "paged":
                keys = torch.cat((past_key_value_state[0], keys), dim=2)
                values = torch.cat((past_key_value_state[1], values), dim=2)
            elif attn_algorithm != "paged":
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
            values_e = values.unsqueeze(2).expand(-1, -1, expansion, -1, -1).flatten(1, 2)
        else:
            keys_e = keys
            values_e = values
        # Save the tokens produced in this forward pass (needed for assign())
        fresh_keys_e = keys_e
        fresh_values_e = values_e

        if attn_algorithm:
            # Pick which fused attn kernels will run.
            use_flash = attn_algorithm == "flash"
            use_mem_efficient = attn_algorithm == "mem"
            use_math = attn_algorithm == "math"
            use_paged = attn_algorithm == "paged"

            if not use_paged:
                torch.backends.cuda.enable_flash_sdp(use_flash)
                torch.backends.cuda.enable_mem_efficient_sdp(use_mem_efficient)
                torch.backends.cuda.enable_math_sdp(use_math)

        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(dtype=queries.dtype)

        # flex_attention: Do the math – a fused kernel that computes Q × Kᵀ → scores, 
        # applies soft-max (with optional mask/score hooks), and multiplies by V.  
        # Optimised to run block-sparse layouts on modern GPUs.

        # PagedAttention: Manage memory – a lightweight page-table & cache manager 
        # for long/variable-length KV caches during autoregressive decoding. It never 
        # multiplies tensors itself; it just maps logical token positions to physical 
        # blocks, allocates/free pages, and then asks flex_attention (or Flash) to do 
        # the computation.
        if attn_algorithm == "paged":
            # ------------------------------------------------------------
            # 1.  Ensure global KV caches are large enough
            # ------------------------------------------------------------
            blk_size = block_size or self.paged_attention_config.get(
                "block_size", _DEFAULT_SPARSE_BLOCK_SIZE
            )
            max_blks = self.paged_attention_config.get("max_blocks", None)
            self._validate_paged_attention(q_len, blk_size, max_blks)
            # ------------------------------------------------------------------
            # Fast path: stateless paged‑attention (no KV cache).
            # If the caller did *not* request caching (`use_cache=False`) and did
            # not supply a prior `past_key_value_state`, we should avoid touching
            # the persistent global KV caches.  Otherwise, a second forward pass
            # on the same module would accidentally attend to tokens from the
            # previous sequence and yield incorrect results (as seen in
            # test_llama_paged_attention).  In this stateless case we simply
            # run flex_attention directly and return the result.
            # ------------------------------------------------------------------
            if not use_cache and past_key_value_state is None:
                attn = _flex_attention(
                    queries,
                    keys_e,
                    values_e,
                    scale=None,
                )
                attn = (
                    attn.transpose(2, 1)
                    .contiguous()
                    .view(batch_size, q_len, self.nheads * self.emb_v_per_head)
                )
                return self.dense(attn)

            # Lazily construct (or rebuild) the manager the first time we need it
            # or whenever the requested page‑size changes.
            if (
                self._paged_mgr is None
                or getattr(self._paged_mgr, "page_size", None) != blk_size
            ):
                num_blocks_per_seq = (q_len + blk_size - 1) // blk_size
                # Keep **one** spare page per sequence so a single incremental
                # token can be cached without reallocating the whole manager.
                pages_per_seq = (
                    max_blks
                    if max_blks is not None
                    else num_blocks_per_seq + 1
                )
                # Enough pages for the whole batch so reserve() never exhausts the pool
                n_pages = pages_per_seq * batch_size

                self._paged_mgr = PagedAttention(
                    n_pages=n_pages,
                    page_size=blk_size,
                    max_batch_size=batch_size,
                    device=q.device,
                )

            total_slots = self._paged_mgr.n_pages * blk_size
            cache_needs_resize = (
                self._k_cache is None
                or self._k_cache.shape[2] < total_slots
                or self._k_cache.shape[1] != self.nheads
            )
            if cache_needs_resize:
                self._k_cache = torch.empty(
                    1,
                    self.nheads,
                    total_slots,
                    self.emb_kq_per_head,
                    device=q.device,
                    dtype=queries.dtype,
                )
                self._v_cache = torch.empty(
                    1,
                    self.nheads,
                    total_slots,
                    self.emb_v_per_head,
                    device=q.device,
                    dtype=values_e.dtype,
                )

            # ------------------------------------------------------------
            # 2.  Reserve pages and write *only the new tokens* into cache
            # ------------------------------------------------------------
            # Determine starting logical position for each batch element
            prev_capacity = self._paged_mgr.capacity.clone()            # (B,)
            positions = torch.arange(q_len, device=q.device)            # (L)

            # Reserve enough space for the total length (prev + new)
            for b in range(batch_size):
                new_total = int(prev_capacity[b] + q_len)
                self._paged_mgr.reserve(
                    torch.tensor([b], device=q.device),
                    torch.tensor(new_total, device=q.device),
                )

            batch_idx = torch.arange(batch_size, device=q.device)       # (B,)
            # input_pos = prev_capacity.unsqueeze(1) + positions[None, :]
            input_pos = prev_capacity.unsqueeze(1) + positions          # (B, L)

            with torch.no_grad():
                self._paged_mgr.assign(
                    batch_idx,
                    input_pos.to(dtype=torch.int64),
                    fresh_keys_e.detach(),
                    fresh_values_e.detach(),
                    self._k_cache,
                    self._v_cache,
                )

            # ------------------------------------------------------------
            # 3.  Gather per‑batch K/V from the global cache and compute attention
            # ------------------------------------------------------------
            def _gather_kv(b: int):
                # Gather *all* logical indices from 0 .. current_length‑1 so the
                # query token can attend to the full prefix, matching baseline.
                full_len = int(prev_capacity[b] + q_len)
                logical_idx = torch.arange(full_len, device=q.device)            # (full_len,)
                phys_block = self._paged_mgr.page_table[b, (logical_idx // blk_size)]
                offset = phys_block * blk_size + (logical_idx % blk_size)
                k_b = self._k_cache[0, :, offset, :]    # (H, full_len, D)
                v_b = self._v_cache[0, :, offset, :]
                return k_b, v_b

            k_list, v_list = zip(*[_gather_kv(b) for b in range(batch_size)])
            k_src = torch.stack(k_list)   # (B, H, L, D)
            v_src = torch.stack(v_list)

            attn = _flex_attention(
                queries,
                k_src,
                v_src,
                scale=None,
            )

            # (B, H, L, D)  →  (B, L, H*D)
            attn = (
                attn.transpose(2, 1)
                .contiguous()
                .view(batch_size, q_len, self.nheads * self.emb_v_per_head)
            )
            out = self.dense(attn)

            if use_cache:
                return out, (keys, values)
            return out
        else:
            attn = F.scaled_dot_product_attention(
                queries,
                keys_e,
                values_e,
                attn_mask=attn_mask,
                dropout_p=self.p_dropout if self.training else 0.0,
                is_causal=is_causal_mask,
                scale=None,
            )

        if attn_algorithm and not use_paged:
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
        block_size: Optional[int] = None,
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
            block_size,
        )

        # if use_cache=True, we return the hidden_state as well as the kv cache.
        # We only reduce the output, and keep the cache thread-local
        if use_cache:
            out = reduce_from_tensor_model_parallel_region(out_par[0], self.group)
            return out, out_par[1]
        else:
            out = reduce_from_tensor_model_parallel_region(out_par, self.group)
            return out
