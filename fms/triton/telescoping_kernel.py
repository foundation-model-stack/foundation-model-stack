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
    num_groups: tl.constexpr,  # n_heads / kv_heads
    D,  # head_dim
    N,  # buffer len
    C,  # cache size
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
    """ """

    pid_b = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m, pid_n = col_major(
        pid,
        L * H * num_groups,
        C,
        block_size_m,
        block_size_n,
    )

    # A matrix pointers
    offs_k = tl.arange(0, block_size_k)
    offs_b = (pid_b * block_size_b + tl.arange(0, block_size_b)) % B
    offs_am = (pid_m * block_size_m + tl.arange(0, block_size_m)) % (L * H * num_groups)
    a_offs = (
        offs_b[:, None, None] * stride_ab
        + offs_am[None, :, None] * stride_am
        + offs_k[None, None, :] * stride_ak
    )
    a_ptrs = a_ptr + a_offs

    # M matrix pointers
    offs_ml = (pid_m * block_size_l + tl.arange(0, block_size_l)) // H
    offs_mc = (pid_n * block_size_n + tl.arange(0, block_size_n)) % C
    m_offs = offs_ml[:, None] * stride_ml + offs_mc[None, :] * stride_mc
    m_ptrs = m_ptr + m_offs
    n_idxs = tl.load(m_ptrs, mask=offs_ml[:, None] < L, other=N)  # [2, 64]
    n_mask = tl.reshape(n_idxs < N, (1, block_size_l // H, H, 1, block_size_n))

    # B matrix pointers
    offs_bh = tl.arange(0, H)
    b_offs = (
        offs_b[:, None, None, None, None] * stride_bb  # Batch offsets # [64, 64]
        + tl.reshape(n_idxs, (1, block_size_l // H, H, 1, block_size_n))
        * stride_bn  # Index values
        + offs_bh[None, None, :, None, None] * stride_bh  # Index values
        + offs_k[None, None, None, :, None] * stride_bk  # k block
    )
    b_ptrs = b_ptr + b_offs
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[block_m, block_n]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros(
        (block_size_b * block_size_l, num_groups, block_size_n), dtype=tl.float32
    )

    for k in range(0, tl.cdiv(D, block_size_k)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        a = tl.reshape(
            tl.load(
                a_ptrs,
                mask=offs_k[None, None, :] < D - k * block_size_k,
                other=0.0,
            ),
            (block_size_b * block_size_l, num_groups, block_size_k),
        )
        b = tl.reshape(
            tl.load(
                b_ptrs,
                mask=n_mask
                & (offs_k[None, None, None, :, None] < D - k * block_size_k),
                other=0.0,
            ),
            (block_size_b * block_size_l, block_size_k, block_size_n),
        )
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += block_size_k * stride_ak
        b_ptrs += block_size_k * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_om = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_on = pid_n * block_size_n + tl.arange(0, block_size_n)
    o_ptrs = (
        o_ptr
        + offs_b[:, None, None] * stride_ob
        + offs_om[None, :, None] * stride_om
        + offs_on[None, None, :] * stride_on
    )
    o_mask = (offs_om[None, :, None] < L * H * num_groups) & (
        offs_on[None, None, :] < C
    )
    tl.store(
        o_ptrs,
        tl.reshape(accumulator, (block_size_b, block_size_m, block_size_n)),
        mask=o_mask,
    )


def invoke_telescoping_kernel(
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

    assert gs * h <= config["block_size_m"]

    output = torch.zeros((bs, sl, h, gs, cs), dtype=A.dtype).cuda()

    grid = lambda META: (
        triton.cdiv(bs, META["block_size_b"]),
        triton.cdiv(sl * h * gs, META["block_size_m"])
        * triton.cdiv(cs, META["block_size_n"]),
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
        block_size_l=block_size_l,
        **config,
    )
    return output


@triton.jit()
def telescoping_bwd_a_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    o_ptr,
    m_ptr,
    # Matrix dimensions
    B,  # batch
    L,  # length
    H: tl.constexpr,  # kv heads
    num_groups: tl.constexpr,  # n_heads / kv_heads
    D,  # head_dim
    N,  # buffer len
    C,  # cache size
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bn,
    stride_bh,
    stride_bc,
    stride_ml,
    stride_md,
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
    pid_b = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    pid_m, pid_n = col_major(
        pid,
        L * H * num_groups,
        C,
        block_size_m,
        block_size_n,
    )

    # A matrix pointers
    offs_k = pid_k * block_size_k + tl.arange(0, block_size_k)
    offs_n = tl.arange(0, block_size_n)

    offs_b = (pid_b * block_size_b + tl.arange(0, block_size_b)) % B
    offs_am = (pid_m * block_size_m + tl.arange(0, block_size_m)) % (L * H * num_groups)
    a_offs = (
        offs_b[:, None, None] * stride_ab
        + offs_am[None, :, None] * stride_am
        + offs_k[None, None, :] * stride_ak
    )

    # breakpoint()
    a_ptrs = a_ptr + a_offs
    offs_ml = (pid_m * block_size_l + tl.arange(0, block_size_l)) // H

    # offs_mk = tl.arange(0, block_size_k)
    m_offs = offs_ml[:, None] * stride_ml + offs_k[None, :] * stride_md
    m_ptrs = m_ptr + m_offs

    # n_idxs = tl.load(m_ptrs, mask=offs_ml[:, None] < L, other=N)  # [2, 64]
    # n_mask = tl.reshape(n_idxs < N, (1, block_size_l // H, H, block_size_k, 1))

    # B matrix pointers
    offs_bh = tl.arange(0, H)
    b_offs = (
        offs_b[:, None, None, None, None] * stride_bb  # Batch offsets # [64, 64]
        + offs_bh[None, None, :, None, None] * stride_bh  # Index values
        + (
            pid_n * block_size_n + offs_n[None, None, None, None, :] * stride_bc
        )  # N block
    )
    accumulator = tl.zeros(
        (block_size_b * block_size_l, num_groups, block_size_n), dtype=tl.float32
    )
    # for k in range(0, tl.cdiv(D, block_size_k)):

    # Load the next block of A and B, generate a mask by checking the K dimension.
    a = tl.reshape(
        tl.load(
            a_ptrs,
            mask=offs_k[None, None, :] < D,  # - k * block_size_k,
            other=0.0,
        ),
        (block_size_b * block_size_l, num_groups, block_size_k),
    )

    # breakpoint()
    n_idxs = tl.load(m_ptrs, mask=offs_ml[:, None] < L, other=N)  # [2, 64]
    n_mask = tl.reshape(n_idxs < N, (1, block_size_l // H, H, block_size_k, 1))

    b_ptrs = (
        b_ptr
        + b_offs
        + tl.reshape(n_idxs, (1, block_size_l // H, H, block_size_k, 1)) * stride_bn
    )  # Index values
    b = tl.reshape(
        tl.load(
            b_ptrs,
            mask=n_mask
            & (offs_k[None, None, None, :, None] < D),  # - k * block_size_k),
            other=0.0,
        ),
        (block_size_b * block_size_l, block_size_k, block_size_n),
    )
    # We accumulate along the K dimension.
    accumulator = tl.dot(a, b)

    # Advance the ptrs to the next K block.
    # a_ptrs += block_size_k * stride_ak
    # m_ptrs += block_size_k * stride_md

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_om = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_on = pid_n * block_size_n + tl.arange(0, block_size_n)
    o_ptrs = (
        o_ptr
        + offs_b[:, None, None] * stride_ob
        + offs_om[None, :, None] * stride_om
        + offs_on[None, None, :] * stride_on
    )

    o_mask = (offs_om[None, :, None] < L * H * num_groups) & (
        offs_on[None, None, :] < C
    )

    # breakpoint()
    tl.atomic_add(
        o_ptrs,
        tl.reshape(accumulator, (block_size_b, block_size_m, block_size_n)),
        mask=o_mask,
    )


def invoke_telescoping_bwd_a_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    M: torch.Tensor,
    config: dict,
):
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert M.is_contiguous()

    bs, sl, h, gs, d = A.shape
    _, n, _, cs = B.shape
    _, _ = M.shape

    assert gs * h <= config["block_size_m"]

    output = torch.zeros((bs, sl, h, gs, cs), dtype=A.dtype).cuda()

    grid = lambda META: (
        triton.cdiv(bs, META["block_size_b"]),
        triton.cdiv(sl * h * gs, META["block_size_m"])
        * triton.cdiv(cs, META["block_size_n"]),
        triton.cdiv(d, META["block_size_k"]),
    )

    total_blocks_bs = triton.cdiv(bs, config["block_size_b"])
    total_blocks_m = triton.cdiv(sl * h * gs, config["block_size_m"])
    total_blocks_n = triton.cdiv(cs, config["block_size_n"])

    print(f"{total_blocks_bs=}")
    print(f"{total_blocks_m=}")
    print(f"{total_blocks_n=}")

    block_size_l = config["block_size_m"] // gs
    telescoping_bwd_a_kernel[grid](
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
        block_size_l=block_size_l,
        **config,
    )
    return output


@triton.jit()
def telescoping_bwd_b_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    padded_indices_per_block_ptr,
    value_block_mapping_ptr,
    total_padded_indices_ptr,
    o_ptr,
    # Matrix dimensions
    B,  # batch
    L,  # length
    H: tl.constexpr,  # kv heads
    E: tl.constexpr,  # n_heads / kv_heads
    D: tl.constexpr,  # head_dim
    N,  # buffer len
    C,  # cache size
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_ab,
    stride_al,
    stride_ah,
    stride_ae,
    stride_ad,
    stride_bb,
    stride_bl,
    stride_bh,
    stride_be,
    stride_bc,
    stride_ob,
    stride_on,
    stride_oh,
    stride_od,
    # Meta-parameters
    block_size_b: tl.constexpr,
    block_size_m: tl.constexpr,
    block_size_e: tl.constexpr,
):
    """ """

    # For each n: get all n' (l,c) that hold it
    # For each (l,c): slice A (bhen'd), slice B (bhen')
    # Combo-sum slices (bhd)

    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_e = tl.program_id(axis=2)

    total_padded_indices = tl.load(total_padded_indices_ptr)

    if pid_m * block_size_m >= total_padded_indices:
        return

    padded_indices_per_block_ptrs = (
        padded_indices_per_block_ptr + pid_m * block_size_m + tl.arange(0, block_size_m)
    )
    m_idxs = tl.load(padded_indices_per_block_ptrs)
    m_idxs_mask = m_idxs < L * C

    # Get A slice
    # tl.device_print("H", H)
    offs_b = (pid_b * block_size_b + tl.arange(0, block_size_b)) % B
    offs_l = m_idxs // C
    offs_h = tl.arange(0, H)
    # offs_e = tl.arange(0, E)
    offs_d = tl.arange(0, D)
    a_ptrs = (
        a_ptr
        + offs_b[:, None, None, None] * stride_ab
        + offs_l[None, None, None, :] * stride_al
        + offs_h[None, :, None, None] * stride_ah
        # + offs_e[None, None, None, None, :] * stride_ae
        + pid_e * block_size_e * stride_ae
        + offs_d[None, None, :, None] * stride_ad
    )

    # Get B slice
    offs_c = m_idxs % C
    offs_pad = tl.arange(0, 16)
    b_ptrs = (
        b_ptr
        + offs_b[:, None, None, None] * stride_bb
        + offs_l[None, None, :, None] * stride_bl
        + offs_h[None, :, None, None] * stride_bh
        # + offs_e[None, None, None, :, None] * stride_be
        + pid_e * block_size_e * stride_ae
        + offs_c[None, None, :, None] * stride_bc
        + offs_pad[None, None, None, :]
    )

    # Mul/add
    a = tl.reshape(
        tl.load(
            a_ptrs,
            mask=m_idxs_mask[None, None, None, :],
            other=0.0,
        ),
        (block_size_b * H, D, block_size_m),
    )
    # tl.static_print(b_ptrs.shape, m_idxs_mask.shape)
    b = tl.reshape(
        tl.load(
            b_ptrs,
            mask=(offs_pad[None, None, None, :] == 0)
            & m_idxs_mask[None, None, :, None],
            other=0.0,
        ),
        (block_size_b * H, block_size_m, 16),
    )
    # We accumulate along the K dimension.
    # tl.device_print("a", a)
    # tl.device_print("b", b)
    # o_block = tl.sum(a * b[:, :, :, None], 2)  # b' z_h d'
    o_block = tl.dot(a, b)

    # Write O block
    value_idx = tl.load(value_block_mapping_ptr + pid_m)
    o_ptrs = (
        o_ptr
        + value_idx * stride_on
        + offs_b[:, None, None, None] * stride_ob
        + offs_h[None, :, None, None] * stride_oh
        + offs_d[None, None, :, None] * stride_od
        + offs_pad
    )
    tl.atomic_add(
        o_ptrs,
        o_block.reshape(block_size_b, H, D, 16),
        mask=(offs_pad[None, None, None, :] == 0),
    )


def invoke_telescoping_bwd_b_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    M: torch.Tensor,
    G: torch.Tensor,
    config: dict,
):
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert M.is_contiguous()
    assert G.is_contiguous()

    b, l, h, e, d = A.shape
    _, n, _, _ = B.shape
    _, c = M.shape
    print(b, l, h, e, d, n, c)

    output = torch.zeros((b, n, h, d), dtype=A.dtype, device=A.device)

    (
        padded_indices_per_block,
        value_block_mapping,
        total_padded_indices,
    ) = invert_mapping_gpu(M, n, config["block_size_m"])

    grid = lambda META: (
        triton.cdiv(b, META["block_size_b"]),
        triton.cdiv(padded_indices_per_block.shape[0], META["block_size_m"]),
        triton.cdiv(e, META["block_size_e"]),
    )

    telescoping_bwd_b_kernel[grid](
        A,
        G,
        padded_indices_per_block,
        value_block_mapping,
        total_padded_indices,
        output,
        b,
        l,
        h,
        e,
        d,
        n,
        c,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        A.stride(3),
        A.stride(4),
        G.stride(0),
        G.stride(1),
        G.stride(2),
        G.stride(3),
        G.stride(4),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        **config,
    )
    return output


def invert_mapping_gpu(
    M: torch.Tensor, num_unique_values: int, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        M: torch.Tensor - An L x C map to N indices
    Returns:
        D: torch.Tensor - A
        Di: torch.Tensor -
    """

    # First count how many tokens go to each expert
    cnts = torch.zeros(M.shape[0], num_unique_values, dtype=M.dtype, device=M.device)
    ones = torch.ones_like(M, dtype=torch.long)
    cnts.scatter_add_(1, M, ones)
    indices_per_value = cnts.sum(dim=0)

    # Then pad the amount to the block size of work
    indices_per_value_block_padded = (
        torch.floor_divide(indices_per_value + block_size - 1, block_size) * block_size
    )

    # Count how many tokens in total are we computing in the GPU
    cumsum = indices_per_value_block_padded.cumsum(0)
    total_padded_indices = cumsum[-1]

    # For allocation purposes, compute the worst case scenario for how
    # many tokens we could be doing work for if the RNG gods are bad
    max_total_padded_indices = (
        (M.numel() + num_unique_values * (block_size - 1))
        if M.numel() > num_unique_values
        else (M.numel() + 1) * block_size
    )

    # Compute what MoE expert corresponds to each block of work
    # A block of work consists of tokens that all go to the same expert
    # There are total_padded_tokens // block_size blocks of work, but to
    # simplify kernel launches, we allocate based on worst case and use
    # max_total_padded_tokens instead.
    value_block_mapping = torch.zeros(
        max(
            (max_total_padded_indices + block_size - 1) // block_size + 1,
            num_unique_values,
        ),
        dtype=M.dtype,
        device=M.device,
    )

    num_blocks_per_value = cumsum.div(block_size, rounding_mode="floor")
    ones = torch.ones_like(value_block_mapping)
    value_block_mapping.scatter_add_(0, num_blocks_per_value, ones)
    value_block_mapping = value_block_mapping.cumsum(0)

    # Create the mapping between token idxs in the input tensor and
    # the tokens that will go in each work block

    # Count how many pad tokens need adding to the list of tokens in
    # topk_ids, then create a tensor that will fill in the right spots
    # when doing an argsort later to compute padded_token_ids_per_block
    cum_pad_indices = (indices_per_value_block_padded - indices_per_value).cumsum(0)
    padded_indices = torch.zeros(
        max_total_padded_indices - M.numel(),
        dtype=M.dtype,
        device=M.device,
    )
    ones = torch.ones_like(padded_indices)
    padded_indices.scatter_add_(0, cum_pad_indices[:-1], ones)
    padded_indices = padded_indices.cumsum(0)
    padded_indices_per_block = torch.cat([M.view(-1), padded_indices]).argsort()

    return padded_indices_per_block, value_block_mapping, total_padded_indices


class IndLinear(torch.autograd.Function):
    @staticmethod
    def forward(A,B,M):
        config = {
            "block_size_b": 1,
            "block_size_m": 64,
            "block_size_n": 64,
            "block_size_k": 64,
        }
        return invoke_telescoping_kernel(A, B, M, config)

    @staticmethod
    def setup_context(ctx, inputs, output):
        A,B,M = inputs
        ctx.save_for_backward(A,B,M)

    @staticmethod
    def backward(ctx, G):
        A,B,M = ctx.saved_tensors

        config_a = {
            "block_size_b": 1,
            "block_size_m": 64,
            "block_size_n": 64,
            "block_size_k": 64,
        }
        A_grad = invoke_telescoping_bwd_a_kernel(G, B, M, config_a)

        config_b = {
            "block_size_b": 1,
            "block_size_m": 16,
            "block_size_e": 1,
        }
        B_grad = invoke_telescoping_bwd_b_kernel(A, B, M, G, config_b)

        return A_grad, B_grad, None


class IndLinearTransposed(torch.autograd.Function):
    @staticmethod
    def forward(A,B,M):
        config = {
            "block_size_b": 1,
            "block_size_m": 64,
            "block_size_n": 64,
            "block_size_k": 64,
        }
        return invoke_telescoping_bwd_a_kernel(A, B, M, config)

    @staticmethod
    def setup_context(ctx, inputs, output):
        A,B,M = inputs
        ctx.save_for_backward(A,B,M)

    @staticmethod
    def backward(ctx, G):
        A,B,M = ctx.saved_tensors

        config_a = {
            "block_size_b": 1,
            "block_size_m": 64,
            "block_size_n": 64,
            "block_size_k": 64,
        }
        A_grad = invoke_telescoping_kernel(G, B, M, config_a)

        config_b = {
            "block_size_b": 1,
            "block_size_m": 16,
            "block_size_e": 1,
        }
        B_grad = invoke_telescoping_bwd_b_kernel(G, B, M, A, config_b)

        return A_grad, B_grad, None


b = 1
l = 4096
n = 8192
h = 2
e = 16
c = 128
d = 128

A = torch.randn(b, l, h, e, d, requires_grad=True, device="cuda")
B = torch.randn(b, n, h, d, requires_grad=True, device="cuda")
M = torch.rand(l, c).mul(n).long().cuda()

@torch.compile
def f(A, B, M, b, l, h, d, c):
    Z = (
        B.unsqueeze(-1)
        .expand(-1, -1, -1, -1, c)
        .gather(1, M.view(1, l, 1, 1, c).expand(b, -1, h, d, -1))
    )
    O = A.matmul(Z)
    return O

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CUDA,
        torch.profiler.ProfilerActivity.CPU,
    ],
    with_stack=True,
) as prof:
    torch.cuda.memory._record_memory_history(max_entries=100000)
    Z = (
        B.unsqueeze(-1)
        .expand(-1, -1, -1, -1, c)
        .gather(1, M.view(1, l, 1, 1, c).expand(b, -1, h, d, -1))
    )
    O = A.matmul(Z)
    loss = O.pow(2).sum()
    loss.backward()

    A2 = torch.empty_like(A, requires_grad=True)
    A2.data = A.data
    B2 = torch.empty_like(B, requires_grad=True)
    B2.data = B.data
    indlinear = IndLinear.apply
    O2 = inlinear(A2,B2,M)
    loss2 = O2.pow(2).sum()
    loss2.backward()

    print(
        A.grad.sub(A2.grad).abs().mean(), A.grad.sub(A2.grad).abs().mean() / A.grad.abs().mean()
    )
    print(
        B.grad.sub(B2.grad).abs().mean(), B.grad.sub(B2.grad).abs().mean() / B.grad.abs().mean()
    )

prof.export_chrome_trace("./trace.json")
torch.cuda.memory._dump_snapshot("./memory.pickle")



# Check correctness of transposed fn
A = torch.randn(b, l, h, e, c, requires_grad=True, device="cuda")
B = torch.randn(b, n, h, d, requires_grad=True, device="cuda")
M = torch.rand(l, c).mul(n).long().cuda()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CUDA,
        torch.profiler.ProfilerActivity.CPU,
    ],
    with_stack=True,
) as prof:
    torch.cuda.memory._record_memory_history(max_entries=100000)
    Z = (
        B.unsqueeze(-2)
        .expand(-1, -1, -1, c, -1)
        .gather(1, M.view(1, l, 1, c, 1).expand(b, -1, h, -1, d))
    )
    O = A.matmul(Z)
    loss = O.pow(2).sum()
    loss.backward()
    
    A2 = torch.empty_like(A, requires_grad=True)
    A2.data = A.data
    B2 = torch.empty_like(B, requires_grad=True)
    B2.data = B.data
    indlinear = IndLinearTransposed.apply
    O2 = inlinear(A2,B2,M)
    loss2 = O2.pow(2).sum()
    loss2.backward()
    
    print(
        A.grad.sub(A2.grad).abs().mean(), A.grad.sub(A2.grad).abs().mean() / A.grad.abs().mean()
    )
    print(
        B.grad.sub(B2.grad).abs().mean(), B.grad.sub(B2.grad).abs().mean() / B.grad.abs().mean()
    )

prof.export_chrome_trace("./trace2.json")
torch.cuda.memory._dump_snapshot("./memory2.pickle")
