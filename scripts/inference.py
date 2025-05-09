import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch._inductor.config
from torch import distributed as dist

from fms.models import get_model
from fms.utils import generation, tokenizers
from fms.utils.generation import generate, pad_input_ids


# This example script validates the LLaMA implementation by running inference on a couple of prompts.
#
# Example usage with single-GPU 7B model on slurm, with torch.compile and determinstic behavior:
# CUBLAS_WORKSPACE_CONFIG=:4096:8 srun -N 1 --gres=gpu:1 python scripts/inference.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --compile --deterministic
# Example usage of 13B model on 2 GPUs with Tensor Parallel:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 scripts/inference.py --model_path=~/models/13B-F --tokenizer=~/models/tokenizer.model --distributed
# Example usage with Ring Attention:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 scripts/inference.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --distributed --distributed_strategy=ring

parser = argparse.ArgumentParser(
    description="Script to run inference on a causal model"
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
    default=None,
    help="The model variant (configuration) to benchmark. E.g. 7b, 13b, 70b.",
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--model_source",
    type=str,
    help="Source of the checkpoint. E.g. 'meta', 'hf', None",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    required=True,
    help="Path to the tokenizer (e.g. ~/tokenizer.model)",
)
parser.add_argument(
    "--no_use_cache",
    dest="use_cache", # Store True by default, action stores False
    action="store_false",
    help="Disable the kv-cache (on by default)",
)
parser.add_argument(
    "--unfuse_weights",
    action="store_true",
    help="If set to True, this will unfuse any fused weight modules that support the unfuse_weights method",
)
parser.add_argument(
    "--default_dtype",
    type=str,
    default=None,
    choices=["bf16", "fp16", "fp32"],
    help="If set to one of the choices, overrides the model checkpoint weight format by setting the default pytorch format",
)
parser.add_argument(
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
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
    help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    help="This is a distributed job (multiple instances run with RANK+WORLD_SIZE)",
)
parser.add_argument(
    "--distributed_strategy",
    type=str,
    default=None,
    choices=["tp", "mp", "fsdp", "hsdp", "ddp", "ring"], # Added 'ring'
    help="The distributed strategy to use. If None, will attempt to guess based on --distributed and device count.",
)
parser.add_argument(
    "--batch_input",
    action="store_true",
    help="use a batch of prompts as input",
)
parser.add_argument(
    "--min_pad_length",
    type=int,
    help="Pad inputs to a minimum specified length. If any prompt is larger than the specified length, padding will be determined by the largest prompt",
    default=0,
)
parser.add_argument("--context_file", type=str, default=None, help="File to summarize")

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

default_dtype = None
dtypes_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
if args.default_dtype is not None:
    default_dtype = dtypes_map[args.default_dtype]

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)  # pytorch random seed
    np.random.seed(SEED)  # numpy random seed
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()

print("loading model")

# Determine the distributed strategy
distr_strategy = args.distributed_strategy # Get from command line first

# If not provided via command line, try to infer
if distr_strategy is None:
    if args.distributed:
        distr_strategy = "tp"  # Default to TP if --distributed is set and no strategy specified
    elif torch.cuda.device_count() > 1 and world_size == 1:
        distr_strategy = "mp"  # Default to MP on single node multi-GPU if not distributed
    # Note: 'ring' is not inferred automatically, must be specified.

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    device_type=args.device_type,
    source=args.model_source,
    distributed_strategy=distr_strategy,
    # Pass group only if distributed is initialized
    group=dist.group.WORLD if args.distributed else None,
    fused_weights=not args.unfuse_weights,
    data_type=default_dtype # Pass parsed dtype
)

tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
print("loading complete on rank", local_rank)

if args.compile:
    print("compiling model")
    # compiling can make first inference pass slow
    model.compile(mode=args.compile_mode)


def ids_for_prompt(prompt):
    tokens = tokenizer.tokenize(prompt)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = [tokenizer.bos_token_id] + ids
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return ids


if args.context_file is not None:
    # during testing, the context_file used was a copy/paste of the text of:
    # https://arxiv.org/pdf/2306.15595.pdf
    with open(args.context_file) as file:
        long_prompt = file.read()
        prompt1 = (
            long_prompt
            + "\nPlease give me a brief summary of this research paper in a few bullet points."
        )
        # prompt1 = long_prompt + "\nDescribe work that was done concurrently with the research in this paper."
        prompt2 = long_prompt + "\nPlease write me the abstract for this paper."
else:
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

    prompt1 = template.format(
        "Provide a list of instructions for preparing chicken soup."
    )
    prompt2 = template.format("Explain some popular greetings in Spanish.")

prompt1 = ids_for_prompt(prompt1)
prompt2 = ids_for_prompt(prompt2)
max_len = max([len(prompt) for prompt in [prompt1, prompt2]])


if args.batch_input:
    ids = [prompt1, prompt2]
    ids, padding_kwargs = pad_input_ids(ids, min_pad_length=args.min_pad_length)
else:
    ids = prompt1
    if args.min_pad_length != 0:
        ids, padding_kwargs = pad_input_ids([ids], min_pad_length=args.min_pad_length)
    else:
        padding_kwargs = None


def print_result(result):
    if local_rank != 0:
        return
    if padding_kwargs is not None:
        result = generation.trim_prefix(result)

    result = generation.trim_prefix(result, tokenizer.bos_token_id)

    # stop at EOS token if present and remove padding
    result = generation.truncate_after_eos(result, tokenizer.eos_token_id)

    output_str = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(result)
    )

    print(output_str)
    print()


def infer(use_cache, do_sample):
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    if local_rank == 0:
        print("use_cache", use_cache, ";; do_sample", do_sample)
        print("==================")
    if (
        getattr(model.config, "ntk_scaling", None) is not None
        and model.config.ntk_scaling
    ):
        max_seq_len = max(max_len, model.config.max_expected_seq_len)
    else:
        # without ntk scaling, extending the seq length too far gives bogus results.
        max_seq_len = model.config.max_expected_seq_len

    # Pass padding_kwargs if they exist
    current_extra_kwargs = padding_kwargs if padding_kwargs else {}

    result = generate(
        model,
        ids,
        max_new_tokens=100,
        use_cache=use_cache,
        do_sample=do_sample,
        max_seq_len=max_seq_len,
        extra_kwargs=current_extra_kwargs,
    )
    if len(result.shape) == 1:
        result = result.unsqueeze(0)

    for i in range(result.shape[0]):
        print_result(result[i])


print("generating output", local_rank)
do_sample = [False]
use_cache_settings = [args.use_cache] # Use the argument value directly

for sample, cache in itertools.product(do_sample, use_cache_settings):
    # RingAttention currently does not support use_cache, skip if necessary
    if distr_strategy == 'ring' and cache:
        if local_rank == 0:
            print("Skipping use_cache=True with RingAttention strategy as it is not supported.")
        continue
    infer(cache, sample)
