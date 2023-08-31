import argparse
import itertools
import os

import torch

from fms.models.llama import LLaMA, load_fms_llama
from fms.utils.generation import generate


# This example script validates the LLaMA implementation by running inference on a couple of prompts.

parser = argparse.ArgumentParser(description="Script to run inference on a LLaMA model")
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the directory containing LLaMa weights",
)
parser.add_argument(
    "--tokenizer", type=str, required=True, help="Path to the tokenizer (e.g. ~/tokenizer.model)"
)
parser.add_argument(
    "--compile", type=bool, default=False, help="Use torch.compile (slow for first inference pass)"
)
parser.add_argument(
    "--deterministic", type=bool, default=False, help="Set torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`"
)

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
device = torch.device(args.device_type, local_rank)

torch.set_default_device(device)
torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    torch.use_deterministic_algorithms(True)

print("loading model")
model, tokenizer = load_fms_llama(args.model_path, args.tokenizer)
model.eval()

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
prompt2 = pad_prompt(prompt2, max_len)
ids = torch.stack((prompt2, prompt1), dim=0)

def print_result(result):
    # stop at EOS token if present
    eos_idx = torch.where(result == tokenizer.convert_tokens_to_ids("</s>"))
    eos_idx = eos_idx[0]
    if eos_idx.shape[0] >= 1:
        eos_idx = eos_idx[0].item()
        result = result[: eos_idx + 1]

    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result)))


def infer(use_cache, do_sample):
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    print("use_cache", use_cache, ";; do_sample", do_sample)
    print("==================")
    result = generate(
        model, ids.clone().detach(), max_new_tokens=50, use_cache=use_cache, do_sample=do_sample
    )
    for i in range(result.shape[0]):
        print_result(result[i])
        print()


print("generating output")
do_sample = [True, False]
use_cache = [True, False] # these are identical with greedy iff `torch.use_deterministic_algorithms(True)`
for sample, cache in itertools.product(do_sample, use_cache):
    infer(cache, sample)
    print()
    print()
