import argparse
import os
import statistics
import time
import timeit

import torch
from torch import distributed as dist

from fms.models.llama import load_fms_llama
from fms.utils import generation, print0


# Example running llama 7B on one A100:
#
# $ srun -N 1 --gres=gpu:1 torchrun --nproc_per_node=1 ./scripts/benchmark_inference.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --batch_size=2 --seq_len=500
# loading model
# loading complete on rank 0
# Uncompiled results:
# - with use_cache=True, excluding first call
#         35.20 ms per token
# - without cache
#         91.87 ms per token
# Compiling model...
#
# Compiled results:
# - with use_cache=True, excluding first call
#         21.31 ms per token
# - without cache
#         72.23 ms per token


parser = argparse.ArgumentParser(
    description="Script to benchmark inference time per token on a LLaMA model"
)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
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
    default=100,
    help="Sequence length of mock input",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Batch size of mock input",
)

parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
device = torch.device(args.device_type, local_rank)

torch.set_default_device(device)
torch.set_default_dtype(torch.half)

if args.distributed:
    dist.init_process_group()

print("loading model")
model, tokenizer = load_fms_llama(args.model_path, args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
print("loading complete on rank", local_rank)

SEQ_LEN = args.seq_len
BATCH_SIZE = args.batch_size

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

iters = 25
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
def one_token(m, use_cache):
    if use_cache:
        actual, _ = m.forward(
            next_val, past_key_value_states=cache, use_cache=True, only_last_token=True
        )
        actual = torch.argmax(actual, dim=-1)
        if local_rank == 0:
            torch.testing.assert_close(actual, expected)
    else:
        actual = m.forward(next_input, only_last_token=True)
        actual = torch.argmax(actual, dim=-1)
        if local_rank == 0:
            torch.testing.assert_close(actual, expected)


def end_to_end(model, use_cache, expected=None):
    result = generation.generate(
        model, ids, max_new_tokens=iters, do_sample=False, use_cache=use_cache
    )
    if local_rank == 0:
        assert (
            result.size()[-1] == SEQ_LEN + iters
        ), f"{result.size()}, {SEQ_LEN}, {iters}"
    if expected is not None:
        torch.testing.assert_close(result, expected)
    return result


def log_result(result):
    if local_rank == 0:
        median = statistics.median(result)
        per_token = median / iters
        ms = per_token * 1000
        print(f"\t{ms:0.2f} ms per token")


def bench_one(use_cache):
    print0(f"- with use_cache={use_cache}")
    log_result(
        timeit.repeat(lambda: one_token(model, use_cache), number=iters, repeat=repeat)
    )


def bench_end_to_end(use_cache):
    e2e_expected = end_to_end(model, use_cache)
    print0(f"- with use_cache={use_cache}")
    result = timeit.repeat(
        lambda: end_to_end(model, use_cache, e2e_expected), number=1, repeat=repeat
    )
    log_result(result)


print0("Uncompiled results:")

bench_one(True)
bench_one(False)

print0("End-to-end sequence generation")
bench_end_to_end(True)
bench_end_to_end(False)

print0("Compiling model...")

# Ideally this should be:
# model = torch.compile(model, mode="reduce-overhead", dynamic=True)
model = torch.compile(model)

# Warmup. Especially with torch.compile, first inference pass can be slow.
one_token(model, True)
one_token(model, False)

print0("Compiled results:")

# These get much better results with mode='reduce-overhead' but can lead to
# some memory issues
bench_one(True)
bench_one(False)

# Each new token changes the sequence length, leading to "guard failures,"
# making this slow. This should be improved with `dynamic=True`
# print0()
# print0("(Compiled) End-to-end sequence generation")
# bench_end_to_end(True)
# bench_end_to_end(False)
