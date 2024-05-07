import pathlib
from typing import Tuple

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


"""Fused MoE kernel."""


@triton.jit()
def col_major(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    pid_m = pid % grid_m
    pid_n = pid // grid_m
    return pid_m, pid_n


@triton.jit()
def fused_moe_kernel_v3(
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
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    compute_type: tl.constexpr,
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
    by expert index and padding ensures divisibility by block_m, which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert.
    """

    pid = tl.program_id(axis=0)
    pid_m, pid_n = col_major(
        pid,
        EM,
        N,
        block_m,
        block_n,
    )

    num_tokens_post_padded = tl.load(total_padded_tokens_ptr)

    if pid_m * block_m >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * block_m + tl.arange(0, block_m)
    offs_token = tl.load(padded_token_ids_per_block_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_bn = (pid_n * block_n + tl.arange(0, block_n)) % N
    offs_k = tl.arange(0, block_k)
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
    # We accumulate into a `[block_m, block_n]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, block_k)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * block_k),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * block_k, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(compute_type), mask=c_mask)


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
    assert B.is_contiguous()
    assert C.is_contiguous()

    EM = padded_token_ids_per_block.shape[0]
    N = B.shape[1]

    grid = lambda META: (
        triton.cdiv(EM, META["block_m"]) * triton.cdiv(N, META["block_n"]),
    )

    compute_type = tl.float16 if A.dtype == torch.float16 else tl.bfloat16
    if A.device.type == "cpu":
        compute_type = tl.float32
    fused_moe_kernel_v3[grid](
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
        compute_type=compute_type,
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
    return best, best_config


def _load_best_configs():
    saved_configs = pathlib.Path.cwd() / "moe_mm_configs.p"
    if saved_configs.is_file():
        import pickle

        with open(saved_configs, "rb") as f:
            return pickle.load(f)


def _save_best_configs(best_configs):
    saved_configs = pathlib.Path.cwd() / "moe_mm_configs.p"
    with open(saved_configs, "wb") as f:
        import pickle

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
        import functools

        # TODO: Add more configs?
        configs = [
            {
                "block_m": 64,
                "block_n": 64,
                "block_k": 32,
            },
            {
                "block_m": 16,
                "block_n": 32,
                "block_k": 64,
            },
        ]

        configs = [config for config in configs if config["block_m"] == padding_size]
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


@torch.library.impl(lib, "moe_mm", "CPU")
def moe_mm_cpu(
    input: torch.Tensor,
    moe_matrix: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    topk: int,
    padding_size,
):
    T, D = input.shape
    M, A = token_expert_mapping.shape

    a = input.view(T, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(T * topk, moe_matrix.shape[1], dtype=a.dtype, device=a.device)

    token_expert_mapping = token_expert_mapping.view(-1)
    for i in range(moe_matrix.shape[0]):
        mask = token_expert_mapping == i
        if mask.sum():
            out[mask] = a[mask] @ moe_matrix[i].transpose(0, 1)
    return out.view(M, A, moe_matrix.shape[1])


# TODO: Add a Backward kernel for Mixtral training in the future
