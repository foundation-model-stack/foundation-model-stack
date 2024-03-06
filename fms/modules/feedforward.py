import pathlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from sympy import xfield
from torch.distributed.distributed_c10d import ProcessGroup

from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.tp import TPModule


class FeedForwardBlock(nn.Module):
    """
    A two-layer, symmetric, fully-connected MLP structure.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of input and output vectors.
    hidden_grow_factor : float
        Sets dimensionality of inner latent space (emb_dim * hidden_grow_factor)
    multiple_of : Optional[int]
        Ensure inner latent space is a multiple of this parameter if defined (useful for
        TensorParallel as well as GPU kernel speed)
    activation_fn : nn.Module
        An activation function over torch.FloatTensors applied to inner latent space.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor=4.0,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
        gain=1,
    ):
        super(FeedForwardBlock, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        self.w1 = nn.Linear(emb_dim, self.hidden_dim, bias=use_bias)
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = nn.Linear(self.hidden_dim, emb_dim, bias=use_bias)
        self.use_bias = use_bias
        self.reset_params(gain=gain)

    def reset_params(self, gain=1):
        # Fulfills following constraints in expectation:
        #  - Norm of w1 and w2 are equal (for step-normalizing optimizers like AdamW / Sophia)
        #  - Norm of output equals norm of input times gamma
        # when activation is relu-like
        for layer in ["w1", "w2"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=(2**0.5 * gain / self.w1.weight.numel() ** 0.5) ** 0.5,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def forward(self, x):
        out = self.a(self.w1(x))
        if self.p_dropout:
            out = self.d(out)
        out = self.w2(out)
        return out


class TPFeedForwardBlock(FeedForwardBlock, TPModule):
    """
    A two-layer, symmetric, fully-connected MLP structure with Tensor Parallel support.

    Args
    ----
    Check FeedForwardBlock for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor: float = 4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
        gain=1,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        rank, world_size = distributed.rank_and_world(group)
        assert (
            hidden_dim % world_size == 0
        ), "Hidden dim must be divisible by world size"
        FeedForwardBlock.__init__(
            self,
            emb_dim,
            hidden_grow_factor / world_size,
            multiple_of,
            activation_fn,
            p_dropout,
            use_bias,
            gain,
        )
        self.setup_tp(rank, world_size)

    def colwise_param_names(self) -> List[str]:
        return ["w1"]

    def rowwise_param_names(self) -> List[str]:
        return ["w2"]

    @staticmethod
    def import_module(
        ffb: FeedForwardBlock, group: ProcessGroup
    ) -> "TPFeedForwardBlock":
        tp_ffb = TPFeedForwardBlock(
            emb_dim=ffb.w1.in_features,
            hidden_grow_factor=ffb.hidden_dim / ffb.w1.in_features,
            multiple_of=None,
            activation_fn=ffb.a,
            p_dropout=ffb.p_dropout,
            use_bias=ffb.use_bias,
            group=group,
        )
        return tp_ffb

    def forward(self, x):
        x_par = copy_to_tensor_model_parallel_region(x)
        out_par = FeedForwardBlock.forward(self, x_par)
        return reduce_from_tensor_model_parallel_region(out_par)


class GatedLinearUnit(nn.Module):
    """
    A two-point-five-layer, fully-connected gated linear MLP structure (GLU).
    Contains 50% extra params compared to FeedForwardBlock, adjust accordingly.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of input and output vectors.
    hidden_grow_factor : float
        Sets dimensionality of inner latent space (emb_dim * hidden_grow_factor)
    multiple_of : Optional[int]
        Ensure inner latent space is a multiple of this parameter if defined (useful for
        TensorParallel as well as GPU kernel speed)
    activation_fn : nn.Module
        An activation function over torch.FloatTensors applied to inner latent gates.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor: float = 4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
        gain=1,
    ):
        super(GatedLinearUnit, self).__init__()
        self.hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            self.hidden_dim = multiple_of * (
                (self.hidden_dim + multiple_of - 1) // multiple_of
            )
        self.w1 = nn.Linear(emb_dim, self.hidden_dim, bias=use_bias)
        self.wg = nn.Linear(emb_dim, self.hidden_dim, bias=use_bias)
        self.a = activation_fn
        self.p_dropout = p_dropout
        if p_dropout:
            self.d = nn.Dropout(p_dropout)
        self.w2 = nn.Linear(self.hidden_dim, emb_dim, bias=use_bias)
        self.use_bias = use_bias
        self.width = emb_dim
        self.grow_factor = hidden_grow_factor
        self.reset_params(gain=gain)

    def reset_params(self, gain=1):
        # Fulfills following constraints in expectation:
        #  - Norm of w1, wg and w2 are equal (for step-normalizing optimizers like AdamW / Sophia)
        #  - Norm of output equals norm of input times gamma
        # when activation is relu-like and input is standard normal
        for layer in ["w1", "w2", "wg"]:
            nn.init.trunc_normal_(
                getattr(self, layer).weight,
                mean=0.0,
                std=(2 * gain**2 / self.grow_factor) ** (1 / 6) / self.width**0.5,
            )
            if self.use_bias:
                getattr(self, layer).bias.data.zero_()

    def forward(self, x):
        out = self.a(self.wg(x)) * self.w1(x)
        if self.p_dropout:
            out = self.d(out)
        return self.w2(out)


class TPGatedLinearUnit(GatedLinearUnit, TPModule):
    """
    A two-point-five-layer, fully-connected gated linear MLP structure (GLU).
    Contains 50% extra params compared to FeedForwardBlock, adjust accordingly.
    This subclass adds Tensor Parallel support.

    Args
    ----
    Check GatedLinearUnit for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        emb_dim,
        hidden_grow_factor: float = 4,
        multiple_of=None,
        activation_fn=nn.ReLU(),
        p_dropout=0.1,
        use_bias=True,
        gain=1,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)

        hidden_dim = int(hidden_grow_factor * emb_dim)
        if multiple_of:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert (
            hidden_dim % world_size == 0
        ), "Hidden dim must be divisible by world size"
        GatedLinearUnit.__init__(
            self,
            emb_dim,
            hidden_grow_factor / world_size,
            multiple_of,
            activation_fn,
            p_dropout,
            use_bias,
            gain,
        )
        self.setup_tp(rank, world_size)

    def colwise_param_names(self) -> List[str]:
        return ["w1", "wg"]

    def rowwise_param_names(self) -> List[str]:
        return ["w2"]

    @staticmethod
    def import_module(glu: GatedLinearUnit, group: ProcessGroup) -> "TPGatedLinearUnit":
        tp_glu = TPGatedLinearUnit(
            emb_dim=glu.width,
            hidden_grow_factor=glu.hidden_dim / glu.width,
            multiple_of=None,
            activation_fn=glu.a,
            p_dropout=glu.p_dropout,
            use_bias=glu.use_bias,
            group=group,
        )

        return tp_glu

    def forward(self, x):
        x_par = copy_to_tensor_model_parallel_region(x)
        out_par = GatedLinearUnit.forward(self, x_par)
        return reduce_from_tensor_model_parallel_region(out_par)


"""Fused MoE kernel."""


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    padded_token_ids_per_block_ptr,
    expert_block_mapping_ptr,
    total_padded_tokens_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    top_k: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can be any shape representing batches and K is the feature dimension of each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is the number of experts, K is the input feature dimension, and N is the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the total number of tokens post padding, topk is the number of times each token is repeated,
        and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens, repeated topk times and arranged by the expert index they are assigned to.
    - expert_ids: A tensor containing the indices of the expert for each block. It determines which expert matrix from B should be used for each block in A.
    This kernel performs the multiplication of a token by its corresponding expert matrix as determined by `expert_ids`. The sorting of `sorted_token_ids`
    by expert index and padding ensures divisibility by BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    total_blocks_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(total_padded_tokens_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(padded_token_ids_per_block_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    off_experts = tl.load(expert_block_mapping_ptr + pid_m)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, total_blocks_k):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None]
            & (offs_k[None, :] < K - k * (BLOCK_SIZE_K * SPLIT_K)),
            other=0.0,
        )
        b = tl.load(
            b_ptrs, mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K * SPLIT_K), other=0.0
        )
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * stride_bk * SPLIT_K

    # accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """

    # First count how many tokens go to each expert
    cnts = torch.zeros(
        topk_ids.shape[0], num_experts, dtype=topk_ids.dtype, device=topk_ids.device
    )
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)

    # Then pad the amount to the block size of work
    tokens_per_expert_block_padded = (
        torch.floor_divide(tokens_per_expert + block_size - 1, block_size) * block_size
    )

    # Count how many tokens in total are we computing in the GPU
    cumsum = tokens_per_expert_block_padded.cumsum(0)
    total_padded_tokens = cumsum[-1]

    # For allocation purposes, compute the worst case scenario for how
    # many tokens we could be doing work for if the RNG gods are bad
    max_total_padded_tokens = (
        (topk_ids.numel() + num_experts * (block_size - 1))
        if topk_ids.numel() > num_experts
        else (topk_ids.numel() + 1) * block_size
    )

    # Compute what MoE expert corresponds to each block of work
    # A block of work consists of tokens that all go to the same expert
    # There are total_padded_tokens // block_size blocks of work, but to
    # simplify kernel launches, we allocate based on worst case and use
    # max_total_padded_tokens instead.
    expert_block_mapping = torch.zeros(
        max((max_total_padded_tokens + block_size - 1) // block_size + 1, num_experts),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )

    num_blocks_per_expert = cumsum.div(block_size, rounding_mode="floor")
    ones = torch.ones_like(expert_block_mapping)
    expert_block_mapping.scatter_add_(0, num_blocks_per_expert, ones)
    expert_block_mapping = expert_block_mapping.cumsum(0)

    # Create the mapping between token idxs in the input tensor and
    # the tokens that will go in each work block

    # Count how many pad tokens need adding to the list of tokens in
    # topk_ids, then create a tensor that will fill in the right spots
    # when doing an argsort later to compute padded_token_ids_per_block
    cum_pad_tokens = (tokens_per_expert_block_padded - tokens_per_expert).cumsum(0)
    padded_tokens = torch.zeros(
        max_total_padded_tokens - topk_ids.numel(),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    ones = torch.ones_like(padded_tokens)
    padded_tokens.scatter_add_(0, cum_pad_tokens[:-1], ones)
    padded_tokens = padded_tokens.cumsum(0)
    padded_token_ids_per_block = torch.cat([topk_ids.view(-1), padded_tokens]).argsort()

    return padded_token_ids_per_block, expert_block_mapping, total_padded_tokens


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    top_k: int,
    config: dict,
):
    assert A.is_contiguous()
    assert C.is_contiguous()

    grid = lambda META: (
        triton.cdiv(padded_token_ids_per_block.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
        META["SPLIT_K"],
    )

    fused_moe_kernel[grid](
        A,
        B,
        C,
        padded_token_ids_per_block,
        expert_block_mapping,
        total_padded_tokens,
        B.shape[1],
        B.shape[2],
        padded_token_ids_per_block.shape[0],
        token_expert_mapping.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        top_k=top_k,
        **config,
    )


def _autotune(configs, function):
    import torch.utils.benchmark as benchmark

    def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
        try:
            f(*args, **kwargs)
            t0 = benchmark.Timer(
                stmt="f(*args, **kwargs)",
                globals={"args": args, "kwargs": kwargs, "f": f},
            )
        except:
            return None
        return t0.blocked_autorange().mean * 1e6

    best = None
    best_config = None
    for config in configs:
        t_config = benchmark_torch_function_in_microseconds(function, config)
        if t_config is not None:
            if best is not None:
                if t_config < best:
                    best = t_config
                    best_config = config
            else:
                best = t_config
                best_config = config
        print(str(config), " :", str(t_config))
    return best, best_config


def _load_best_configs():
    saved_configs = pathlib.Path.cwd() / "moe_mm_configs.p"
    if saved_configs.is_file():
        import pickle

        with open(saved_configs, "rb") as f:
            print(f"Loading best configs from file {saved_configs}")
            return pickle.load(f)


def _save_best_configs(best_configs):
    saved_configs = pathlib.Path.cwd() / "moe_mm_configs.p"
    with open(saved_configs, "wb") as f:
        import pickle

        print(f"Saving best configs to file {saved_configs}")
        pickle.dump(best_configs, f)


def _create_best_configs_key(input, moe_matrix, token_expert_mapping):
    key = (input.size(), token_expert_mapping.size(), moe_matrix.size())
    return key


BEST_MOE_CONFIGS = None

lib = torch.library.Library("moe", "FRAGMENT")
lib.define(
    "moe_mm(Tensor input, Tensor moe_matrix, Tensor token_expert_mapping, Tensor padded_token_ids_per_block, Tensor expert_block_mapping, Tensor total_padded_tokens, int topk, int padding_size) -> Tensor"
)


# All that's needed for torch.compile support
@torch.library.impl(lib, "moe_mm", "Meta")
def moe_mm_meta(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
    padding_size,
):
    M, A = token_expert_mapping.shape
    _, N, _ = moe_matrix.shape
    return torch.empty((M, A, N), device=input.device, dtype=input.dtype)


@torch.library.impl(lib, "moe_mm", "CUDA")
def moe_mm(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
    padding_size,
):
    M, A = token_expert_mapping.shape
    E, N, _ = moe_matrix.shape
    output = torch.zeros((M, A, N), device=input.device, dtype=input.dtype)

    global BEST_MOE_CONFIGS
    if BEST_MOE_CONFIGS is None:
        BEST_MOE_CONFIGS = _load_best_configs()
    # Loading must have not been successful. Let's create a new dictionary.
    if BEST_MOE_CONFIGS is None:
        BEST_MOE_CONFIGS = {}
    key = _create_best_configs_key(input, moe_matrix, token_expert_mapping)
    if key not in BEST_MOE_CONFIGS:
        print("key ", key, " not found. Running autotune. This might take a while.")
        import functools

        # TODO: Add more configs?
        configs = [
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 2,
                "num_warps": 8,
                # "num_stages": 4,
            },
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 2,
                "num_warps": 8,
                # "num_stages": 4,
            },
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 2,
                "num_warps": 8,
                # "num_stages": 4,
            },
            {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "SPLIT_K": 4,
                "num_warps": 8,
                # "num_stages": 4,
            },
        ]
        configs = [
            config for config in configs if config["BLOCK_SIZE_M"] == padding_size
        ]
        print("all configs len: ", len(configs))
        best, best_config = _autotune(
            configs,
            functools.partial(
                invoke_fused_moe_kernel,
                input,
                moe_matrix,
                output,
                token_expert_mapping,
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
                topk,
            ),
        )
        BEST_MOE_CONFIGS[key] = best_config
        print("Found best_config ", best_config, " with time ", best, " for key ", key)
        _save_best_configs(BEST_MOE_CONFIGS)
    best_config = BEST_MOE_CONFIGS[key]
    if best_config is None:
        return torch.tensor([])

    invoke_fused_moe_kernel(
        input,
        moe_matrix,
        output,
        token_expert_mapping,
        padded_token_ids_per_block,
        expert_block_mapping,
        total_padded_tokens,
        topk,
        best_config,
    )

    return output


### vLLM kernel for comparison
@triton.jit
def fused_moe_kernel_vllm(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can be any shape representing batches and K is the feature dimension of each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is the number of experts, K is the input feature dimension, and N is the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the total number of tokens post padding, topk is the number of times each token is repeated,
        and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens, repeated topk times and arranged by the expert index they are assigned to.
    - expert_ids: A tensor containing the indices of the expert for each block. It determines which expert matrix from B should be used for each block in A.
    This kernel performs the multiplication of a token by its corresponding expert matrix as determined by `expert_ids`. The sorting of `sorted_token_ids`
    by expert index and padding ensures divisibility by BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    off_experts = tl.load(expert_ids_ptr + pid_m)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel_vllm(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    fused_moe_kernel_vllm[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16,
        **config,
    )


lib.define(
    "moe_mm_vllm(Tensor input, Tensor moe_matrix, Tensor token_expert_mapping, Tensor padded_token_ids_per_block, Tensor expert_block_mapping, Tensor total_padded_tokens, int topk) -> Tensor"
)


# All that's needed for torch.compile support
@torch.library.impl(lib, "moe_mm_vllm", "Meta")
def moe_mm_vllm_meta(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
):
    M, A = token_expert_mapping.shape
    _, N, _ = moe_matrix.shape
    return torch.empty((M, A, N), device=input.device, dtype=input.dtype)


@torch.library.impl(lib, "moe_mm_vllm", "CUDA")
def moe_mm_vllm(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
):
    M, A = token_expert_mapping.shape
    E, N, _ = moe_matrix.shape
    output = torch.zeros((M, A, N), device=input.device, dtype=input.dtype)

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }

    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }

    invoke_fused_moe_kernel_vllm(
        input,
        moe_matrix,
        output,
        torch.empty_like(token_expert_mapping),
        token_expert_mapping,
        padded_token_ids_per_block,
        expert_block_mapping,
        total_padded_tokens,
        False,
        topk,
        config,
    )

    return output


lib.define(
    "align_vllm(Tensor topk_ids, int block_size, int num_experts) -> (Tensor, Tensor, Tensor)"
)


# All that's needed for torch.compile support
@torch.library.impl(lib, "align_vllm", "Meta")
def align_vllm_meta(
    topk_ids,
    block_size,
    num_experts,
):
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (topk_ids.numel() + num_experts,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    return sorted_ids, expert_ids, num_tokens_post_pad


@torch.library.impl(lib, "align_vllm", "CUDA")
def moe_align_block_size_vllm(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    from vllm._C import ops

    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (topk_ids.numel() + num_experts,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    ops.moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


class ConditionalFeedForward(nn.Module):
    """
    This class represents the expert feed forward networks of an MoE FF layer.

    For more information, see the review paper in https://arxiv.org/pdf/2209.01667.pdf

    Args
    ----
    num_experts : int
        The number of expert feed forward networks.
    dim : int
        The embedding dimension for the transformer model.
    intermediate_size : int
        The intermediate size for the expert networks.
    """

    def __init__(self, num_experts: int, dim: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, intermediate_size))
        self.moe_impl = "fms"

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        # if x.shape[0] > 4:
        if self.moe_impl == "fms":
            ## Triton path
            # Check constraints.
            assert x.shape[1] == self.w13.shape[2], "Hidden size mismatch"
            assert x.is_contiguous(), "Hidden_states must be contiguous"
            assert self.w13.is_contiguous(), "Expert weights 1 must be contiguous"
            assert self.w2.is_contiguous(), "Expert weights 2 must be contiguous"

            M, _ = x.shape
            E, N, _ = self.w13.shape

            if expert_indices.numel() <= E:
                padding_size = 16
            else:
                padding_size = 32

            (
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
            ) = torch.ops.moe.align_vllm(expert_indices, padding_size, E)

            x1, x3 = (
                torch.ops.moe.moe_mm(
                    x,
                    self.w13,
                    expert_indices,
                    padded_token_ids_per_block,
                    expert_block_mapping,
                    total_padded_tokens,
                    expert_indices.shape[1],
                    padding_size,
                )
                .view(-1, N)
                .chunk(2, dim=1)
            )
            return torch.ops.moe.moe_mm(
                F.silu(x1) * x3,
                self.w2,
                expert_indices,
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
                1,
                padding_size,
            )
        elif self.moe_impl == "vllm":
            # Check constraints.
            assert x.shape[1] == self.w13.shape[2], "Hidden size mismatch"
            assert x.is_contiguous(), "Hidden_states must be contiguous"
            assert self.w13.is_contiguous(), "Expert weights 1 must be contiguous"
            assert self.w2.is_contiguous(), "Expert weights 2 must be contiguous"

            M, _ = x.shape
            E, N, _ = self.w13.shape

            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            }

            if M <= E:
                config = {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 1,
                }

            (
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
            ) = torch.ops.moe.align_vllm(expert_indices, config["BLOCK_SIZE_M"], E)

            x1, x3 = (
                torch.ops.moe.moe_mm_vllm(
                    x,
                    self.w13,
                    expert_indices,
                    padded_token_ids_per_block,
                    expert_block_mapping,
                    total_padded_tokens,
                    expert_indices.shape[1],
                )
                .view(-1, N)
                .chunk(2, dim=1)
            )

            return torch.ops.moe.moe_mm_vllm(
                F.silu(x1) * x3,
                self.w2,
                expert_indices,
                padded_token_ids_per_block,
                expert_block_mapping,
                total_padded_tokens,
                1,
            )

        elif self.moe_impl == "gpt-fast":
            ## Pure Pytorch path
            # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
            w13_weights = self.w13[expert_indices].transpose(-1, -2)  # [T, A, D, D]
            # w3_weights = self.w13[:, self.intermediate_size:][expert_indices].transpose(-1, -2)  # [T, A, D, D]
            w2_weights = self.w2[expert_indices]  # [T, A, D, D]
            x1, x3 = torch.einsum("ti, taio -> tao", x, w13_weights).chunk(2, dim=2)
            # x3 = torch.einsum("ti, taio -> tao", x, w3_weights)
            expert_outs = torch.einsum(
                "tao, taio -> tai", (F.silu(x1) * x3), w2_weights
            )
            return expert_outs


class TPConditionalFeedForward(ConditionalFeedForward, TPModule):
    """
    This class represents the expert feed forward networks of an MoE FF layer.
    This subclass adds TP support.

    Args
    ----
    num_experts : int
        The number of expert feed forward networks.
    dim : int
        The embedding dimension for the transformer model.
    intermediate_size : int
        The intermediate size for the expert networks.
    """

    def __init__(
        self,
        num_experts: int,
        dim: int,
        intermediate_size: int,
        group: Optional[ProcessGroup] = None,
    ):
        assert torch.distributed.is_initialized()
        rank, world_size = distributed.rank_and_world(group)

        assert (
            intermediate_size % world_size == 0
        ), "Intermediate size must be divisible by world size"
        ConditionalFeedForward.__init__(
            self,
            num_experts,
            dim,
            intermediate_size // world_size,
        )
        self.setup_tp(rank, world_size)

    def moe_param_names(self) -> List[str]:
        return ["w13", "w2"]

    @staticmethod
    def import_module(
        cff: ConditionalFeedForward, group: ProcessGroup
    ) -> "TPConditionalFeedForward":
        tp_cff = TPConditionalFeedForward(
            num_experts=cff.num_experts,
            dim=cff.dim,
            intermediate_size=cff.intermediate_size,
            group=group,
        )

        return tp_cff

    def forward(self, x, expert_indices):
        x_par = copy_to_tensor_model_parallel_region(x)
        out_par = ConditionalFeedForward.forward(self, x_par, expert_indices)
        return reduce_from_tensor_model_parallel_region(out_par)


class MOEFeedForward(nn.Module):
    """
    A Sparse Mixture Of Experts (MoE) Feed Forward layer. The output of this layer for a
    given input is determined by the weighted sum of the outputs of a subset of size
    `num_activated_experts` of the `num_experts` expert networks. The weights are given
    by the gating network, then passed through a topK and a softmax filter to make it _sparse_.

    For more information, see the review paper in https://arxiv.org/pdf/2209.01667.pdf

    Args
    ----
    num_experts : int
        The number of expert feed forward networks.
    num_activated_experts : int
        How many experts can be activated at any single time.
    dim : int
        The embedding dimension for the transformer model.
    intermediate_size : int
        The intermediate size for the expert networks.
    """

    def __init__(
        self,
        num_experts: int,
        num_activated_experts: int,
        dim: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(num_experts, dim, intermediate_size)
        self.dim = dim
        self.num_activated_experts = num_activated_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape[:2]
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # x: [T, D]
        scores = self.gate(x)  # [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(
            expert_weights, self.num_activated_experts, dim=-1
        )  # [T, A], [T, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)  # [T, A]
        # Given the balloning memory requirements, only process at most 10 tokens at a time
        # if x.shape[0] > 10:
        #     split_x = x.chunk(x.shape[0] // 10 + 1)
        #     split_ei = expert_indices.chunk(expert_indices.shape[0] // 10 + 1)
        #     expert_outs = torch.cat([self.cond_ffn(x_i, ei_i) for x_i, ei_i in zip(split_x, split_ei)], dim=0)
        # else:
        expert_outs = self.cond_ffn(x, expert_indices)
        # print(expert_outs.shape)
        int_v1 = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
        int_v2 = int_v1.view(B, S, self.dim)
        return int_v2
