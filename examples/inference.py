import argparse
import itertools
import os

import torch
from torch import distributed as dist

from fms.distributed.strategy import TensorParallelStrategy
from fms.models.llama import LLaMA, load_fms_llama
from fms.utils.generation import generate


# This example script validates the LLaMA implementation by running inference on a couple of prompts.
# Example usage:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 ~/repos/newfms/examples/inference.py --model_path=~/models/13-F --tokenizer=~/models/tokenizer.model --distributed

parser = argparse.ArgumentParser(description="Script to run inference on a LLaMA model")
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
    "--compile",
    action="store_true",
    help="Use torch.compile (slow for first inference pass)",
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

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))


device = torch.device(args.device_type, local_rank)

torch.set_default_device(device)
torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()

print("loading model")
model, tokenizer = load_fms_llama(args.model_path, args.tokenizer)
model.eval()
print("loading complete on rank", local_rank)

if args.compile:
    print("compiling model")
    # compiling can make first inference pass slow
    model = torch.compile(model)

prompt1 = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nProvide a list of instructions for preparing chicken soup.\n\n### Response:"
prompt2 = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain some popular greetings in Spanish.\n\n### Response:"


def ids_for_prompt(prompt):
    tokens = tokenizer.tokenize(prompt)
    tokens = ["<s>"] + tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor(ids, dtype=torch.long, device=device)
    return ids


def pad_prompt(prompt, pad_len, pad_token="<unk>"):
    to_pad = pad_len - len(prompt)
    if to_pad == 0:
        return prompt

    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    pad_ids = [pad_id] * to_pad
    return torch.cat((torch.tensor(pad_ids, device=device), prompt))


prompt1 = ids_for_prompt(prompt1)
prompt2 = ids_for_prompt(prompt2)

max_len = max([len(prompt) for prompt in [prompt1, prompt2]])
prompt1 = pad_prompt(prompt1, max_len)
# LLaMA 7B did better on the spanish prompt vs 13B.
# TODO: add a better english prompt to demonstrate padding/batching.
#prompt2 = pad_prompt(prompt2, max_len)
#ids = torch.stack((prompt2, prompt1), dim=0)
ids = prompt1.unsqueeze(0)

def print_result(result):
    if local_rank != 0:
        return
    # stop at EOS token if present
    eos_idx = torch.where(result == tokenizer.convert_tokens_to_ids("</s>"))
    eos_idx = eos_idx[0]
    if eos_idx.shape[0] >= 1:
        eos_idx = eos_idx[0].item()
        result = result[: eos_idx + 1]

    # print(result)
    # print(tokenizer.convert_ids_to_tokens(result))
    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result)))
    print()


def infer(use_cache, do_sample):
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    if local_rank == 0:
        print("use_cache", use_cache, ";; do_sample", do_sample)
        print("==================")
    result = generate(
        model,
        ids,
        max_new_tokens=50,
        use_cache=use_cache,
        do_sample=do_sample,
    )
    for i in range(result.shape[0]):
        print_result(result[i])


print("generating output", local_rank)
do_sample = [True, False]
use_cache = [True, False]  # True/False are identical with greedy iff `torch.use_deterministic_algorithms(True)`
for sample, cache in itertools.product(do_sample, use_cache):
    dist.barrier()
    infer(cache, sample)

