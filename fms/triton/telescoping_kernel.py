import pathlib
from typing import Any, Dict, Optional, Tuple

import torch
import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]


"""Fused telescoping cache kernel."""


@triton.jit()
def col_major(pid, m, n, block_m: tl.constexpr, block_n: tl.constexpr):
    grid_m = tl.cdiv(m, block_m)
    pid_m = pid % grid_m
    pid_n = pid // grid_m
    return pid_m, pid_n


@triton.jit()
def telescoping_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    o_ptr,
    m_ptr,
    # Matrix dimensions
    B,  # batch
    L,  # length
    H: tl.constexpr,  # kv heads
    num_groups: tl.constexpr, # n_heads / kv_heads
    D, # head_dim
    N, # buffer len
    C, # cache size
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bn,
    stride_bh,
    stride_bk,
    stride_ml,
    stride_mc,
    stride_ob,
    stride_om,
    stride_on,
    # Meta-parameters
    block_size_b: tl.constexpr, 
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
    block_size_l: tl.constexpr,
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

    pid_b = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m, pid_n = col_major(
        pid,
        L*H*num_groups,
        C,
        block_size_m,
        block_size_n,
    )

    # A matrix pointers
    offs_k = tl.arange(0, block_size_k)
    offs_b = (pid_b * block_size_b + tl.arange(0, block_size_b)) % B
    offs_am = (pid_m * block_size_m + tl.arange(0, block_size_m)) % (L*H*num_groups)
    a_offs = offs_b[:, None, None] * stride_ab + offs_am[None, :, None] * stride_am + offs_k[None, None, :] * stride_ak
    a_ptrs = a_ptr + a_offs

    # M matrix pointers
    offs_ml = (pid_m * block_size_l + tl.arange(0, block_size_l)) // H
    offs_mc = (pid_n * block_size_n + tl.arange(0, block_size_n)) % C
    m_offs = offs_ml[:, None] * stride_ml + offs_mc[None, :] * stride_mc
    m_ptrs = m_ptr + m_offs
    n_idxs = tl.load(m_ptrs,
                     mask=offs_ml[:, None] < L,
                     other=N) # [2, 64]
    n_mask = tl.reshape(n_idxs < N, (1, block_size_l // H, H, 1, block_size_n))
    
    # B matrix pointers
    offs_bh = tl.arange(0, H)
    b_offs = (
        offs_b[:, None, None, None, None] * stride_bb # Batch offsets # [64, 64]
        + tl.reshape(n_idxs, (1, block_size_l // H, H, 1, block_size_n)) * stride_bn # Index values
        + offs_bh[None, None, :, None, None] * stride_bh # Index values
        + offs_k[None, None, None, :, None] * stride_bk # k block
    )
    b_ptrs = (
        b_ptr 
        + b_offs
    )
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[block_m, block_n]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((block_size_b * block_size_l, num_groups, block_size_n), dtype=tl.float32)

    for k in range(0, tl.cdiv(D, block_size_k)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.reshape(tl.load(
            a_ptrs,
            mask=offs_k[None, None, :] < D - k * block_size_k,
            other=0.0,
        ), (block_size_b * block_size_l, num_groups, block_size_k))
        b = tl.reshape(tl.load(
            b_ptrs,
            mask=n_mask & (offs_k[None, None, None, :, None] < D - k * block_size_k),
            other=0.0
        ), (block_size_b * block_size_l, block_size_k, block_size_n))
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += block_size_k * stride_ak
        b_ptrs += block_size_k * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_om = (pid_m * block_size_m + tl.arange(0, block_size_m))
    offs_on = (pid_n * block_size_n + tl.arange(0, block_size_n))
    o_ptrs = o_ptr + offs_b[:, None, None] * stride_ob + offs_om[None, :, None] * stride_om + offs_on[None, None, :] * stride_on
    o_mask = (offs_om[None, :, None] < L*H*num_groups) & (offs_on[None, None, :] < C)
    tl.store(o_ptrs, tl.reshape(accumulator, (block_size_b, block_size_m, block_size_n)), mask=o_mask)


def invoke_telescopic_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    M: torch.Tensor,
    config: dict,
):
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert M.is_contiguous()

    bs, sl, h, gs, d = A.shape
    _, n, _, _ = B.shape
    _, cs = M.shape

    assert gs*h <= config["block_size_m"]

    output = torch.zeros((bs, sl, h, gs, cs), dtype=A.dtype).cuda()

    grid = lambda META: (triton.cdiv(bs, META["block_size_b"]),
        triton.cdiv(sl*h*gs, META["block_size_m"]) * triton.cdiv(cs, META["block_size_n"]),
    )

    block_size_l = config["block_size_m"] // gs

    telescoping_kernel[grid](
        A,
        B,
        output,
        M,
        bs,
        sl,
        h,
        gs,
        d,
        n,
        cs,
        A.stride(0),
        A.stride(3),
        A.stride(4),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        B.stride(3),
        M.stride(0),
        M.stride(1),
        output.stride(0),
        output.stride(3),
        output.stride(4),
        block_size_l = block_size_l,
        **config,
    )
    return output


b = 1
l = 4097
n = 8192
h = 2
e = 16
c = 128
d = 128

A = torch.randn(b,l,h,e,d).cuda()
B = torch.randn(b,n,h,d).cuda()
M = torch.rand(l,c).mul(n).long().cuda()

# A = torch.arange(b*l*h*e*d).div(b*l*h*e*d).view(b,l,h,e,d).cuda()
# B = torch.arange(b*n*h*d).div(b*n*h*d).view(b,n,h,d).cuda()
# M = (torch.arange(l*c)*5%17).view(l, c).long().cuda()

@torch.compile
def f(A, B, M, b, l, h, d, c):
    Z = B.unsqueeze(-1).expand(-1,-1,-1,-1,c).gather(1, M.view(1,l,1,1,c).expand(b,-1,h,d,-1))
    O = A.matmul(Z)
    return O

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
) as prof:
    torch.cuda.memory._record_memory_history(max_entries=100000)
    O = f(A, B, M, b, l, h, d, c)

    config = {
        "block_size_b": 1, 
        "block_size_m": 64,
        "block_size_n": 64,
        "block_size_k": 64,
    }
    O_fused = invoke_telescopic_kernel(A, B, M, config)
prof.export_chrome_trace("./trace.json")
torch.cuda.memory._dump_snapshot("./memory.pickle")
print(O.shape, O_fused.shape)
print(O, O_fused)
print(torch.allclose(O, O_fused, atol=1e-2, rtol=0))
print((O-O_fused).abs().max())