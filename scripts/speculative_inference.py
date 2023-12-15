import argparse
import itertools
import os

import torch
from torch import distributed as dist

from fms.models.llama import load_fms_llama
from fms.modules.speculator import Speculator
from fms.utils import generation, tokenizers
from fms.utils.cache import PagedKVCache
from fms.utils.generation import speculative_generate


# This example script validates the LLaMA implementation by running inference on a couple of prompts.
#
# Example usage with single-GPU 7B model on slurm, with torch.compile and determinstic behavior:
# CUBLAS_WORKSPACE_CONFIG=:4096:8 srun -N 1 --gres=gpu:1 python scripts/speculative_inference.py --model_path=~/models/7B-F/ --speculator_path=~/speculator/7B-F/step_6000_ckp.pth --tokenizer=~/models/tokenizer.model --compile --deterministic
# Example usage of 13B model on 2 GPUs with Tensor Parallel:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 scripts/speculative_inference.py --model_path=~/models/13B-F --speculator_path=~/speculator/13B-F/step_6000_ckp.pth --tokenizer=~/models/tokenizer.model --distributed

parser = argparse.ArgumentParser(description="Script to run inference on a LLaMA model")
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the directory containing LLaMa weights (.pth files sharded by tensor parallel rank, not HF weights)",
)
parser.add_argument(
    "--speculator_path",
    type=str,
    required=True,
    help="Path to the checkpoint containing speculator weights (single .pth file, not HF weights)",
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
parser.add_argument("--context_file", type=str, default=None, help="File to summarize")

args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
device = torch.device(args.device_type, local_rank)
torch.cuda.set_device(device)

torch.set_default_device(device)
torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()

print("loading model")
model = load_fms_llama(args.model_path)
tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
print("loading speculator")
speculator = Speculator(model.width, model.config.src_vocab_size, n_predict=3)
speculator.load_state_dict(
    torch.load(args.speculator_path, map_location=device)["model_state"]
)
torch.set_grad_enabled(False)
print("loading complete on rank", local_rank)
print("initiating cache")
kv_cache = PagedKVCache(
    model.config.nlayers,
    model.config.nheads,
    model.config.emb_dim,
    total_num_gpu_blocks=3818,
    dtype=model.shared.emb.weight.dtype,
)

if args.compile:
    print("compiling model")
    # Bug with kv-cache in PT2.1
    torch._inductor.config.joint_graph_constant_folding = False
    # compiling can make first inference pass slow
    model = torch.compile(model, mode=args.compile_mode)
    speculator = torch.compile(speculator, mode=args.compile_mode)



def ids_for_prompt(prompt):
    tokens = tokenizer.tokenize(prompt)
    tokens = ["<s>"] + tokens
    ids = tokenizer.convert_tokens_to_ids(tokens)
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

# ids = [[prompt1]]
# ids = [[prompt1], [prompt2], [prompt2, prompt1]]
ids = [[prompt2, prompt1]]

def print_result(result, inp, n_steps):
    if local_rank != 0:
        return
    # stop at EOS token if present
    result = generation.truncate_after_eos(
        result, tokenizer.convert_tokens_to_ids("</s>")
    )
    # print(result)
    # print(tokenizer.convert_ids_to_tokens(result))
    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(result)))
    print(f"{len(result) - len(inp)} tokens in {n_steps} steps")
    print()


def infer(ids):
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    if local_rank == 0:
        print("==================")
    if model.config.ntk_scaling:
        max_seq_len = max(max_len, model.config.max_expected_seq_len)
    else:
        # without ntk scaling, extending the seq length too far gives bogus results.
        max_seq_len = model.config.max_expected_seq_len

    result, n_steps = speculative_generate(
        model,
        ids,
        speculator,
        new_tokens=100,
        max_seq_len=max_seq_len,
        paged_kv_cache=kv_cache,
    )
    for i in range(len(result)):
        print_result(result[i], ids[i], n_steps)


print("generating output", local_rank)
for inp in ids:
    infer(inp)
