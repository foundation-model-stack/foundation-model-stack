# Flash/paged CPU rewrite of `spyre::paged_attn_compute_with_sinks`

Date: 2026-07-08
Branch: `swa_granite`
File: `fms/utils/spyre/paged.py`

## Problem

The decode custom op `spyre::paged_attn_compute_with_sinks` currently, for each
sequence, gathers every valid KV token from the paged cache into two Python
lists and `torch.stack`s them into full contiguous per-sequence tensors:

```python
keys  = torch.stack(keys_lst,  dim=0)   # [seq_len_kv, kv_heads, head_size]
values = torch.stack(values_lst, dim=0)  # [seq_len_kv, kv_heads, head_size]
```

It then runs a single full attention (einsum QK -> softmax -> einsum V) over
those materialized tensors, with an attention-sink post-scale.

We want a **flash/paged**-style CPU implementation (in the spirit of
`test-spyre-scripts/test_paged.py`'s `paged_cpu`) that iterates the KV
dimension in blocks with an online softmax, so the large contiguous
`[seq_len_kv, H, D]` tensors are never materialized.

## Goal & constraints

- **Primary motivation: memory footprint.** Never hold a full per-sequence
  `[seq_len_kv, H, D]` K or V tensor; only one `block_size` chunk resident at a
  time, plus the `(Sq, H, D)` output accumulator (same size as the output slice,
  unavoidable).
- **Numerically identical (within fp tolerance)** to the current op, including:
  - attention-sink handling via the existing `sigmoid(lse - sinks)` post-scale,
  - the causal + sliding-window mask rule,
  - the "cut pads for first prefill" behaviour,
  - GQA/MQA expansion.
- **The op's external contract is unchanged**: same signature, same
  `register_fake`, same registration in `register_attention_op`. Only the
  internal computation of `paged_attn_compute_with_sinks` changes.

## Approach

Iterate the KV dimension **by cache block** (using `block_table` + offset to
gather one `block_size` chunk from the cache), accumulating with an online
(flash) softmax.

### Per-sequence algorithm (replaces the body's inner work, lines ~182-248)

For each sequence `i` with valid token range `[start_pos, seq_len)`:

1. `seq_len_kv = seq_len - start_pos`. Pad-cut exactly as today: if
   `q.shape[0] > seq_len_kv`, set `seq_len_q_i = seq_len_kv` and `q = q[-seq_len_kv:]`.
2. Online-softmax accumulators, fp32 for stability:
   - `m` — running max of masked logits, shape `(H, Sq)`, init `-inf`.
   - `l` — running denominator, shape `(H, Sq)`, init `0`.
   - `acc` — running weighted V sum, shape `(Sq, H, D)`, init `0`.
3. Walk KV in `block_size`-sized chunks ordered by absolute position `j` from
   `start_pos` to `seq_len`. For each chunk `[j0, j1)`:
   - Gather `k_blk`, `v_blk` for just those positions from the cache using
     `block_number = block_table[j // block_size]`, `offset = j % block_size`.
     One chunk resident; no full stack. (A chunk may straddle at most the tail;
     within a contiguous run of the same block the gather is a slice.)
   - GQA/MQA expand the chunk via `repeat_interleave` (small, per-chunk).
   - `scores = einsum("qhd,khd->hqk", q * sqrt_scale, k_blk * sqrt_scale)`
     giving `(H, Sq, blk)`, where `sqrt_scale = sqrt(scale)` (identical to the
     current op, applied to both q and k).
   - Build the chunk mask from **absolute positions**: causal
     (`key_abs <= query_abs`) plus, when `0 < sliding_window`, sliding-window
     (`query_abs - key_abs < sliding_window`). Additive `-inf`/`0`. This
     reproduces exactly the global mask the current op slices, restricted to the
     chunk's key columns.
   - Online-softmax update:
     - `m_new = maximum(m, rowmax(scores + mask))`  (over the chunk's key dim)
     - `alpha = exp(m - m_new)`
     - `p = exp(scores + mask - m_new)`  (`(H, Sq, blk)`)
     - `l = l * alpha + p.sum(-1)`
     - `acc = acc * alpha_broadcast + einsum("hqk,khd->qhd", p, v_blk)`
     - `m = m_new`
4. After the loop: `lse = m + log(l)` per `(H, Sq)` — equals the current op's
   `torch.logsumexp(QK + mask, dim=-1)`. `attn = acc / l` (broadcast).
5. Sink post-scale, identical to current op:
   `sink_scale = sigmoid((lse - sinks.view(-1, 1)).float())` -> `(H, Sq)`,
   transpose/unsqueeze to `(Sq, H, 1)`, `attn = attn * sink_scale`.
   Write `output[i][-seq_len_q_i:] = attn`.

### Why this is numerically equivalent

- **Softmax/attention**: block-wise online softmax with running max is the
  standard flash-attention identity; it computes the same `softmax(QK+mask) @ V`
  as the current single-shot einsum.
- **LSE**: `m + log(l)` after the full pass equals
  `logsumexp` over all masked logits, so the sink scale is bit-for-bit the same
  formula as today.
- **Mask**: the current op constructs `mask[Sq, seq_len_kv]` as causal tril
  intersected with the sliding-window tril, keyed on positions within
  `[start_pos, seq_len)`. Reconstructing the same predicate per chunk from
  absolute `j` yields identical mask columns. Verified by mask-diff in the test.

Expected numerical difference from the current op comes only from fp accumulation
order (online vs single-shot). fp32 accumulators keep this well within tolerance.

## Verification (before any commit)

Standalone CPU script: `test-spyre-scripts/test_paged_sinks_flash.py`.

- Keeps a **verbatim copy of the current op body** as a local reference oracle
  (`_paged_sinks_reference`), so the comparison is self-contained and does not
  depend on git history.
- Builds random `key_cache`/`value_cache`, `query`, `block_table`,
  `current_tkv_mask`, `left_padded_prompt_mask`, `sinks`.
- Sweeps cases:
  - `sliding_window` disabled (0), small (< seq_len_kv), and >= seq_len_kv,
  - GQA and MQA (num_kv_heads = num_query_heads, and a ratio > 1),
  - a prefill-with-pad case where `q.shape[0] > seq_len_kv`,
  - `start_pos > 0` (left padding).
- Asserts `torch.testing.assert_close(reference, flash)` starting at
  `atol=rtol=1e-3` and tightening. Also asserts reconstructed masks match the
  reference mask for at least one case.

Only after the script passes do we commit.

## Out of scope

- The non-sink op `paged_attn_compute` (unchanged).
- Any Spyre-device kernel / `torch.compile` path — this is the CPU op body only.
- Behavioural changes to masking, sinks, or the store op.
