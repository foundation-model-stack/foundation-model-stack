"""Verify the flash/paged rewrite of spyre::paged_attn_compute_with_sinks.

The reference oracle `_paged_sinks_reference` below is a VERBATIM copy of the
original (materializing) op body -- it stacks all valid KV tokens per sequence
into full contiguous tensors and runs a single-shot attention. We compare it
against the real registered op (which we rewrite to be block-wise / online
softmax) across a sweep of sliding-window sizes, GQA/MQA ratios, prefill-with-
pad, and left-padding cases. Pure CPU; no Spyre device required.
"""

import math
import os
import sys

# Ensure we import the `fms` package from THIS repo (the parent of this
# script's directory), not a shadowing checkout that may sit earlier on the
# path depending on the current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

# Import the module for its side effect of registering the custom op.
import fms.utils.spyre.paged  # noqa: F401

assert os.path.abspath(fms.utils.spyre.paged.__file__).startswith(_REPO_ROOT), (
    f"imported the wrong fms: {fms.utils.spyre.paged.__file__} "
    f"(expected under {_REPO_ROOT})"
)


def _paged_sinks_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale: float,
    current_tkv_mask: torch.Tensor,
    left_padded_prompt_mask: torch.Tensor,
    block_table: torch.Tensor,
    sinks: torch.Tensor,
    sliding_window: int,
) -> torch.Tensor:
    """Verbatim copy of the ORIGINAL materializing op body (the oracle)."""
    output = torch.zeros_like(query)
    num_query_heads = query.shape[2]
    num_kv_heads = value_cache.shape[2]
    head_size = value_cache.shape[3]
    block_size = value_cache.shape[1]
    seq_len_q = query.shape[1]
    num_seqs = query.shape[0]

    block_tables_lst = block_table.tolist()

    seq_lens_lst = current_tkv_mask.tolist()
    for i in range(num_seqs):
        q = query[i]
        block_table_i = block_tables_lst[i]
        start_pos = int(left_padded_prompt_mask[i].item())
        seq_len = int(seq_lens_lst[i])
        seq_len_q_i = seq_len_q

        keys_lst: list[torch.Tensor] = []
        values_lst: list[torch.Tensor] = []
        for j in range(start_pos, seq_len):
            block_number = int(block_table_i[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, block_offset, :, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, block_offset, :, :]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        seq_len_kv = keys.shape[0]

        # cut the pads for first prefill
        if q.shape[0] > seq_len_kv:
            seq_len_q_i = seq_len_kv
            q = q[-seq_len_kv:]

        if num_kv_heads > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_query_heads // num_kv_heads, dim=1)
            values = torch.repeat_interleave(
                values, num_query_heads // num_kv_heads, dim=1
            )

        # Generate mask for prefix attention
        mask = torch.ones((1, seq_len_q_i, seq_len_kv), dtype=torch.bool)
        mask[:, :, -seq_len_q_i:] = torch.tril(mask[:, :, -seq_len_q_i:])
        mask = torch.where(mask.logical_not(), -torch.inf, 0.0).to(
            device=query.device, dtype=query.dtype
        )
        # truncate for sliding window kv_cache
        mask = mask[..., -seq_len_q_i:, -seq_len_kv:]
        if 0 < sliding_window < seq_len_kv:
            mask += torch.tril(
                mask.new_full((seq_len_q_i, seq_len_kv), -torch.inf),
                diagonal=(seq_len_kv - seq_len_q_i) - sliding_window,
            )

        sqrt_scale = math.sqrt(
            scale if scale is not None else 1.0 / math.sqrt(head_size)
        )
        attn_weights = torch.einsum("qhd,khd->hqk", q * sqrt_scale, keys * sqrt_scale)
        attn_weights += mask
        lse = torch.logsumexp(attn_weights, dim=-1)  # (H, Sq)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = torch.einsum("hqk,khd->qhd", attn_weights, values)  # (Sq, H, D)

        sink_scale = torch.sigmoid((lse - sinks.view(-1, 1)).to(torch.float32)).to(
            attn.dtype
        )  # (H, Sq)
        sink_scale_expanded = sink_scale.transpose(0, 1).unsqueeze(-1)  # (Sq, H, 1)
        attn = attn * sink_scale_expanded  # (Sq, H, D)

        output[i][-seq_len_q_i:] = attn
    return output


def _make_case(
    *,
    num_seqs,
    seq_len_q,
    num_kv_heads,
    num_query_heads,
    head_size,
    block_size,
    seq_lens,          # list[int] per-seq total tkv (valid keys end)
    start_positions,   # list[int] per-seq left-pad (valid keys start)
    dtype,
    seed,
):
    gen = torch.Generator().manual_seed(seed)

    max_seq_len = max(seq_lens)
    num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # Give every sequence its own disjoint set of blocks so aliasing can't hide
    # an indexing bug.
    total_blocks = num_seqs * num_blocks_per_seq + 2

    key_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, generator=gen
    )
    value_cache = torch.randn(
        total_blocks, block_size, num_kv_heads, head_size, dtype=dtype, generator=gen
    )
    query = torch.randn(
        num_seqs, seq_len_q, num_query_heads, head_size, dtype=dtype, generator=gen
    )
    sinks = torch.randn(num_query_heads, dtype=dtype, generator=gen)

    block_table = torch.zeros((num_seqs, num_blocks_per_seq), dtype=torch.int64)
    for s in range(num_seqs):
        for b in range(num_blocks_per_seq):
            block_table[s, b] = s * num_blocks_per_seq + b + 1  # avoid block 0

    current_tkv_mask = torch.tensor(seq_lens, dtype=torch.int64)
    left_padded_prompt_mask = torch.tensor(start_positions, dtype=torch.int64)

    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        scale=1.0 / math.sqrt(head_size),
        current_tkv_mask=current_tkv_mask,
        left_padded_prompt_mask=left_padded_prompt_mask,
        block_table=block_table,
        sinks=sinks,
        sliding_window=0,
    )


# fp32: the flash rewrite is algebraically identical to the reference, so in
# fp32 they must agree to ~machine epsilon. This is the strong correctness gate.
# fp16: production dtype -- online (fp32) accumulation vs the reference's
# fp16 softmax-then-matmul legitimately differ by a few fp16 ULPs on rare
# elements, so we allow attention-scale tolerance (still 20x tighter than the
# atol=0.1 that test_paged.py uses for the Spyre device).
_TOL = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=5e-3, rtol=5e-3),
}


def _run_case(name, cfg, sliding_window, head_size, block_size, seed):
    for dtype in (torch.float32, torch.float16):
        case = _make_case(
            head_size=head_size,
            block_size=block_size,
            dtype=dtype,
            seed=seed,
            **cfg,
        )
        case["sliding_window"] = sliding_window

        ref = _paged_sinks_reference(**case)
        got = torch.ops.spyre.paged_attn_compute_with_sinks(**case)

        torch.testing.assert_close(got, ref, **_TOL[dtype])
        max_abs = (got.float() - ref.float()).abs().max().item()
        dt = "fp32" if dtype == torch.float32 else "fp16"
        print(
            f"  [ok] {name:<42} sw={sliding_window:<5} {dt} max|Δ|={max_abs:.3e}"
        )


def main():
    torch.manual_seed(0)

    head_size = 128
    block_size = 64

    # Base geometry knobs shared across cases.
    configs = [
        dict(
            name="MHA decode",
            num_seqs=3,
            seq_len_q=1,
            num_kv_heads=8,
            num_query_heads=8,
            seq_lens=[130, 200, 64],
            start_positions=[0, 5, 0],
        ),
        dict(
            name="GQA decode (ratio 4)",
            num_seqs=2,
            seq_len_q=1,
            num_kv_heads=2,
            num_query_heads=8,
            seq_lens=[150, 70],
            start_positions=[10, 0],
        ),
        dict(
            name="MQA decode (ratio 8)",
            num_seqs=2,
            seq_len_q=1,
            num_kv_heads=1,
            num_query_heads=8,
            seq_lens=[129, 256],
            start_positions=[0, 3],
        ),
        dict(
            name="prefill w/ pad (Sq>kv)",
            num_seqs=2,
            seq_len_q=64,
            num_kv_heads=2,
            num_query_heads=8,
            seq_lens=[40, 50],       # valid kv < seq_len_q -> pad-cut path
            start_positions=[0, 0],
        ),
        dict(
            name="prefill multi-query",
            num_seqs=2,
            seq_len_q=32,
            num_kv_heads=8,
            num_query_heads=8,
            seq_lens=[96, 130],
            start_positions=[8, 0],
        ),
    ]

    seed = 100
    for cfg in configs:
        name = cfg.pop("name")
        seed += 1
        # Sweep sliding-window regimes: disabled, small, and >= kv (no-op).
        min_kv = min(
            int(s) - int(p)
            for s, p in zip(cfg["seq_lens"], cfg["start_positions"])
        )
        for sw in [0, max(1, min_kv // 2), min_kv, min_kv + 100]:
            _run_case(name, cfg, sw, head_size, block_size, seed)

    print("\nAll flash/paged-with-sinks cases match the reference oracle.")


if __name__ == "__main__":
    main()
