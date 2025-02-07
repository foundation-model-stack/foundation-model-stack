import argparse
import logging
import os
import random
import statistics
import timeit

import numpy as np
import torch
from torch import distributed as dist
from torch._dynamo import OptimizedModule

from fms import models
from fms.utils import fusion, generation, print0, tokenizers


# Example running llama 7B on one A100:
#
# (bare metal) $ CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 ./scripts/benchmark_inference.py --architecture=llama --variant=7b --tokenizer=~/models/tokenizer.model --batch_size=2 --seq_len=500
# (slurm) $ srun -N 1 --gres=gpu:1 torchrun --nproc_per_node=1 ./scripts/benchmark_inference.py --architecture=llama --variant=7b --tokenizer=~/models/tokenizer.model --batch_size=2 --seq_len=500
# loading model
# loading complete on rank 0
# Uncompiled results:
# - with use_cache=True
#         34.86 ms per token
# - with use_cache=False
#         86.39 ms per token
# End-to-end sequence generation
# - with use_cache=True
#         37.04 ms per token
# - with use_cache=False
#         90.68 ms per token
# Compiling model...
# Compiled results:
# - with use_cache=True
#         18.66 ms per token
# - with use_cache=False
#         67.66 ms per token

# (Compiled) End-to-end sequence generation
# - with use_cache=True
#         20.61 ms per token
# - with use_cache=False
#         71.45 ms per token


parser = argparse.ArgumentParser(
    description="Script to benchmark inference time per token on a LLaMA model"
)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--architecture",
    type=str,
    default="llama",
    help="The model architecture to benchmark",
)
parser.add_argument(
    "--variant",
    type=str,
    default="7b",
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--seq_len",
    type=int,
    default=512,
    help="Sequence length of mock input",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Batch size of mock input",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=256,
    help="Max number of tokens to generate",
)
parser.add_argument(
    "--compile_mode",
    type=str,
    help="Mode for compilation",
    default="default",
    choices=["default", "reduce-overhead"],
)
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Set seeds and torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument(
    "--skip_correctness_check",
    action="store_true",
    help="Do not test correctness of outputs vs just timing",
)
parser.add_argument(
    "--skip_eager_runs", action="store_true", help="Do not run the eager benchmarks"
)
parser.add_argument(
    "--skip_compile_runs",
    action="store_true",
    help="Do not run the compiled benchmarks",
)
parser.add_argument(
    "--skip_kvcache_runs",
    action="store_true",
    help="Do not run the kv-cache benchmarks",
)
parser.add_argument(
    "--skip_nokvcache_runs",
    action="store_true",
    help="Do not run the no kv-cache benchmarks",
)
parser.add_argument(
    "--skip_single_token_runs",
    action="store_true",
    help="Do not run the single token benchmarks",
)
parser.add_argument(
    "--skip_e2e_runs", action="store_true", help="Do not run the e2e benchmarks"
)
parser.add_argument(
    "--unfuse_weights",
    action="store_true",
    help="If set to True, this will unfuse any fused weight modules that support the unfuse_weights method",
)

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)  # pytorch random seed
    np.random.seed(SEED)  # numpy random seed
    torch.use_deterministic_algorithms(True)

if world_size > 1:
    dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

print("loading model")
model = models.get_model(args.architecture, args.variant, device_type=args.device_type)

if args.unfuse_weights:
    print("unfusing weights")
    model = fusion.apply_unfuse_weights(model)

tokenizer = tokenizers.get_tokenizer(args.tokenizer)

model.eval()
torch.set_grad_enabled(False)
print(f"loading complete on rank {local_rank}")

SEQ_LEN = args.seq_len
BATCH_SIZE = args.batch_size
MAX_NEW_TOKENS = args.max_new_tokens

ids = torch.randint(
    tokenizer.vocab_size(), (BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.long
)

# This first forward call generates the cache for use in cases where
# `use_cache=True`.
#
# For performance purposes, this call can be considered equivalent to
# `use_cache=False`.
#
# The actual performance of generation with `use_cache=True` would be the cost
# of the first token without cache, plus the cost of all subsequent tokens with
# cache. I.e. the amortized per-token cost would depend on the number of tokens
# generated.
logits, cache = model.forward(ids, use_cache=True)
logits = logits[:, -1, :]
next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()
next_input = torch.cat((ids, next_val), dim=-1)

# not still needed
del logits

expected, _ = model.forward(
    next_val, past_key_value_states=cache, use_cache=True, only_last_token=True
)
expected = torch.argmax(expected, dim=-1)

expected2 = model.forward(next_input, only_last_token=True)
expected2 = torch.argmax(expected2, dim=-1)

torch.testing.assert_close(expected, expected2)

repeat = 3


# The function we're measuring, with or without caching.
#
# In a realistic generate function, the sequence length would grow with each
# subsequent token, and so the average cost would be from a variety of sequence
# lengths.
# We capture the time to generate a single token from a given sequence length
# and batch size. This means we're measuring the cost of the forward pass
# in isolation in a way that's easier to compare, and avoids including the cost
# of the concatenation operation.
def one_token(model, use_cache):
    if use_cache:
        actual, _ = model.forward(
            next_val, past_key_value_states=cache, use_cache=True, only_last_token=True
        )
    else:
        actual = model.forward(next_input, only_last_token=True)
    actual = torch.argmax(actual, dim=-1)
    if local_rank == 0 and not args.skip_correctness_check:
        torch.testing.assert_close(actual, expected)
    else:
        if args.device_type == "cuda":
            torch.cuda.synchronize()


def end_to_end(model, use_cache, expected=None):
    result = generation.generate(
        model,
        ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=use_cache,
        contiguous_cache=args.compile_mode == "reduce-overhead"
        and isinstance(
            model, OptimizedModule
        ),  # this is needed for reduce-overhead to work correctly for now
    )
    if local_rank == 0:
        assert result.size()[-1] == SEQ_LEN + MAX_NEW_TOKENS, (
            f"{result.size()}, {SEQ_LEN}, {MAX_NEW_TOKENS}"
        )
    if expected is not None and not args.skip_correctness_check:
        torch.testing.assert_close(result, expected)
    else:
        if args.device_type == "cuda":
            torch.cuda.synchronize()
    return result


e2e_expected_cache = end_to_end(model, True)
e2e_expected_nocache = end_to_end(model, True)


def log_result(result):
    if local_rank == 0:
        median = statistics.median(result)
        per_token = median / MAX_NEW_TOKENS
        ms = per_token * 1000
        print(f"\t{ms:0.2f} ms per token")


def bench_one(use_cache):
    print0(f"- with use_cache={use_cache}")
    log_result(
        timeit.repeat(
            lambda: one_token(model, use_cache), number=MAX_NEW_TOKENS, repeat=repeat
        )
    )


def bench_end_to_end(use_cache, expected):
    print0(f"- with use_cache={use_cache}")
    result = timeit.repeat(
        lambda: end_to_end(model, use_cache, expected), number=1, repeat=repeat
    )
    log_result(result)


print0(
    f"Results for batch size {BATCH_SIZE}, sequence length {SEQ_LEN}, new tokens generated {MAX_NEW_TOKENS}"
)
if not args.skip_eager_runs:
    print0("Uncompiled results:")
    print0("==========")
    if not args.skip_single_token_runs:
        print0("Single token generation")
        if not args.skip_kvcache_runs:
            bench_one(True)
        if not args.skip_nokvcache_runs:
            bench_one(False)

    if not args.skip_e2e_runs:
        print0("End-to-end sequence generation")
        if not args.skip_kvcache_runs:
            bench_end_to_end(True, e2e_expected_cache)
        if not args.skip_nokvcache_runs:
            bench_end_to_end(False, e2e_expected_nocache)

if not args.skip_compile_runs:
    print0("Compiling model...")

    # This is to prevent a bug in PT 2.1 that has been fixed in PT 2.2 nightlies
    torch._inductor.config.joint_graph_constant_folding = False
    # with mode='reduce-overhead' we see better performance but on multi-GPU models
    # hit an error on the end-to-end test below when run after other tests (if it's
    # run first it works, confirming a memory leak):
    # `RuntimeError: Expected curr_block->ptr == block_state.ptr to be true, but got false.`
    model = torch.compile(model, dynamic=True, mode=args.compile_mode)

    print0()
    print0("Compiled results:")
    print0("==========")

    if not args.skip_single_token_runs:
        # Warmup. Especially with torch.compile, first inference pass can be slow.
        print(f"Warming up the compiled model for single token in rank {local_rank}")
        # Activate dynamo logs to ensure some output during compilation
        torch._logging.set_logs(dynamo=logging.INFO)
        if not args.skip_kvcache_runs:
            one_token(model, True)
        if not args.skip_nokvcache_runs:
            one_token(model, False)
        print(f"Model has warmed up in rank {local_rank}")

        # These get much better results with mode='reduce-overhead' but can lead to
        # some memory issues
        print0("(Compiled) Single token generation")
        if not args.skip_kvcache_runs:
            bench_one(True)
        if not args.skip_nokvcache_runs:
            bench_one(False)

    if not args.skip_e2e_runs:
        print0()
        print(f"Warming up the compiled model e2e in rank {local_rank}")
        if not args.skip_kvcache_runs:
            end_to_end(model, True, e2e_expected_cache)
        if not args.skip_nokvcache_runs:
            end_to_end(model, False, e2e_expected_nocache)
        print(f"Model has warmed up e2e in rank {local_rank}")

        print0("(Compiled) End-to-end sequence generation")
        if not args.skip_kvcache_runs:
            bench_end_to_end(True, e2e_expected_cache)
        if not args.skip_nokvcache_runs:
            bench_end_to_end(False, e2e_expected_nocache)
