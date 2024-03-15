import random
from typing import List, Optional, Tuple

import pytest
import torch
#from xformers import ops as xops
#from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
#from vllm.utils import get_max_shared_memory_bytes

import attention_triton 
import attention_cuda
import time


VERSION = ["triton_v1","cuda_v1","cuda_v2"]
TEST_REPETITIONS = 4 
TEST_WARMUP_REPETITIONS= 1
NUM_BLOCKS = [8, 16, 32] # 8, 16, 32
MAX_SEQ_LEN = [96, 2*1024, 4*1024, 8*1024]#, 16*1024]
PARTITION_SIZE = 512
NUM_SEQS = [4, 64, 128, 256] # Arbitrary values for testing
#NUM_PREFILL_SEQS = [128]  # Arbitrary values for testing
NUM_QKV_HEADS = [(32,32), (40, 40), (64, 64)]#, (128, 128)]  # Arbitrary values for testing
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 8]
DTYPES = [torch.half]#, torch.bfloat16, torch.float32]
USE_ALIBI = [False]
SEEDS = [0]
VERIFY_RESULTS = [True]

def create_kv_caches(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = head_size**-0.5
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape,
                                dtype=dtype,
                                device='cuda')
        key_cache.uniform_(-scale, scale)
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches = []
    for _ in range(num_layers):
        value_cache = torch.empty(size=value_cache_shape,
                                  dtype=dtype,
                                  device='cuda')
        value_cache.uniform_(-scale, scale)
        value_caches.append(value_cache)
    return key_caches, value_caches

#@pytest.fixture()
#def kv_cache_factory():
#    return create_kv_caches

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(context_len, device="cuda").int()
            alibi_bias = (position_ids - context_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)




@pytest.mark.parametrize("version", VERSION)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("num_seqs", NUM_SEQS)
@pytest.mark.parametrize("max_seq_len", MAX_SEQ_LEN)
@pytest.mark.parametrize("num_qkv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("verify_results", VERIFY_RESULTS)
def test_paged_attention(
    #kv_cache_factory,
    version: str,
    num_blocks: int,
    num_seqs: int,
    max_seq_len: int,
    num_qkv_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    verify_results: bool
) -> None:

    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_device(3)

    elapsed_time = 0

    for rep in range(TEST_REPETITIONS+TEST_WARMUP_REPETITIONS):
        #recreate data
        scale = float(1.0 / (head_size**0.5))
        num_query_heads, num_kv_heads = num_qkv_heads
        query = torch.empty(size=(num_seqs,
                            num_query_heads,
                            head_size),
                            dtype=dtype,
                            device="cuda")
        query.uniform_(-scale, scale)

        assert num_query_heads % num_kv_heads == 0
        num_queries_per_kv = num_query_heads // num_kv_heads
        head_mapping = torch.repeat_interleave(
            torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
            num_queries_per_kv)
        alibi_slopes = None
        if use_alibi:
            alibi_slopes = torch.randn(num_query_heads,
                                    dtype=torch.float,
                                    device="cuda")

        context_lens = [random.randint(1, max_seq_len) for _ in range(num_seqs)]
        context_lens[-1] = max_seq_len
        context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

        # Create the block tables.
        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        block_tables = []
        for _ in range(num_seqs):
            block_table = [
                random.randint(0, num_blocks - 1)
                for _ in range(max_num_blocks_per_seq)
            ]
            block_tables.append(block_table)
        block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

        # Create the KV caches.
        key_caches, value_caches = create_kv_caches(num_blocks, block_size, 1,
                                                    num_kv_heads, head_size, dtype,
                                                    seed)
        key_cache, value_cache = key_caches[0], value_caches[0]

        # Call the paged attention kernel.
        output = torch.empty_like(query)
        start = time.perf_counter() 
        if version == "triton_v1":
            attention_triton.paged_attention_triton_v1(
                    output=output,
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    #head_mapping,
                    scale=scale,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    block_size=block_size,
                    #alibi_slopes,
                    num_seqs=num_seqs,
                    num_query_heads=num_query_heads,
                    max_seq_len=max_seq_len,
                    max_num_blocks_per_seq=max_num_blocks_per_seq,
                    head_size=head_size)
        elif version == "triton_v2":
            attention_triton.paged_attention_triton_v2(
                    output=output,
                    query=query,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    #head_mapping,
                    scale=scale,
                    block_tables=block_tables,
                    context_lens=context_lens,
                    block_size=block_size,
                    partition_size=PARTITION_SIZE,
                    #max_seq_len,
                    #alibi_slopes,
                    num_seqs=num_seqs,
                    num_query_heads=num_query_heads,
                    max_seq_len=max_seq_len,
                    max_num_blocks_per_seq=max_num_blocks_per_seq,
                    head_size=head_size)
        elif version == "cuda_v1":
            attention_cuda.paged_attention_v1(
                    output,
                    query,
                    key_cache,
                    value_cache,
                    head_mapping,
                    scale,
                    block_tables,
                    context_lens,
                    block_size,
                    max_seq_len,
                    alibi_slopes)
        elif version == "cuda_v2":
            num_partitions = ((max_seq_len + PARTITION_SIZE - 1) //
                            PARTITION_SIZE)
            assert PARTITION_SIZE % block_size == 0
            num_seqs, num_heads, head_size = output.shape
            tmp_output = torch.empty(
                size=(num_seqs, num_heads, num_partitions, head_size),
                dtype=output.dtype,
                device=output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs, num_heads, num_partitions),
                dtype=torch.float32,
                device=output.device,
            )
            max_logits = torch.empty_like(exp_sums)

            attention_cuda.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                head_mapping,
                scale,
                block_tables,
                context_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
            )
        else:
            raise AssertionError(f"Unknown version: {version}")
        # Run tests and increment time for all repetitions
        torch.cuda.synchronize()
        rep_time = time.perf_counter() - start
        #print(f"{version:<10}\t{rep} {(1000*rep_time):.8f} ")
        if rep > TEST_WARMUP_REPETITIONS-1:
            elapsed_time += rep_time
 
    
    # Run the reference implementation.
    correct="NA"
    abs_max_err="NA"
    rel_max_err="NA"
    if verify_results:
        ref_output = torch.empty_like(query)
        ref_single_query_cached_kv_attention(
            ref_output,
            query,
            num_queries_per_kv,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            scale,
            alibi_slopes,
        )
        correct = torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)
        abs_output=torch.abs(output)
        abs_ref_output=torch.abs(ref_output)
        abs_max_err = torch.max(torch.abs(abs_output-abs_ref_output))
        rel_max_err = torch.max(torch.abs(abs_output-abs_ref_output)/abs_ref_output)

        
    msec_str = f"{(1000*elapsed_time)/TEST_REPETITIONS:.3f}"
    print(f"{version:<10}\t{msec_str} ({TEST_REPETITIONS} reps) correct: {correct}")
    print(f"#,{version},{max_seq_len},{max_seq_len},{PARTITION_SIZE},{num_blocks}, \
          {num_seqs},{max_seq_len},{num_query_heads},{num_kv_heads},{head_size}, \
          {block_size},{dtype},{msec_str}, {abs_max_err},{rel_max_err}")
    if not correct:
            print(f"\t{version} expected output:\n\t", ref_output.reshape(-1)[:32])
            print(f"\t{version} incorrect output:\n\t", output.reshape(-1)[:32])




if __name__ == "__main__":
    for version in VERSION:
        test_paged_attention(
            version = version,
            num_blocks= NUM_BLOCKS[0],
            num_seqs = NUM_SEQS[0],
            max_seq_len= MAX_SEQ_LEN[0],
            num_qkv_heads = NUM_QKV_HEADS[0],
            head_size = HEAD_SIZES[0],
            use_alibi = USE_ALIBI[0],
            block_size = BLOCK_SIZES[0],
            dtype = DTYPES[0],
            seed = SEEDS[0],
            verify_results = VERIFY_RESULTS[0]
        )
            


