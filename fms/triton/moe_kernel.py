import torch
from torch.library import wrap_triton
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


"""Fused MoE kernel."""


@triton.jit()
def col_major(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    pid_m = pid % grid_m
    pid_n = pid // grid_m
    return pid_m, pid_n


def filter_padding_size(configs, named_args, **kwargs):
    if "padding_size" in named_args or "padding_size" in kwargs:
        padding_size = named_args.get("padding_size", kwargs.get("padding_size", None))
        return [
            config for config in configs if config.kwargs["block_m"] == padding_size
        ]


# These are the configs we found to work best for Mixtral 8x7b and A100 GPUs
@triton.autotune(
    configs=[
        triton.Config(
            kwargs={
                "block_m": 64,
                "block_n": 64,
                "block_k": 32,
            }
        ),
        triton.Config(
            kwargs={
                "block_m": 16,
                "block_n": 32,
                "block_k": 64,
            }
        ),
    ],
    key=["N", "K", "EM", "num_valid_tokens"],
    prune_configs_by={"early_config_prune": filter_padding_size},
)
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
    padding_size: tl.constexpr,
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


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    token_expert_mapping: torch.Tensor,
    padded_token_ids_per_block: torch.Tensor,
    expert_block_mapping: torch.Tensor,
    total_padded_tokens: torch.Tensor,
    top_k: int,
    padding_size: int,
):
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()

    EM = padded_token_ids_per_block.shape[0]
    N = B.shape[1]

    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["block_m"]) * triton.cdiv(N, META["block_n"]),
    )

    compute_type = tl.float16 if A.dtype == torch.float16 else tl.bfloat16
    if A.device.type == "cpu":
        compute_type = tl.float32
    wrap_triton(fused_moe_kernel_v3)[grid](
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
        padding_size=padding_size,
    )


# TODO: Add a Backward kernel for Mixtral training in the future
