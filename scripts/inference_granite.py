import argparse
import itertools
import os
from typing import Mapping

import torch
import torch._inductor.config
from torch import distributed as dist

from fms import models
from fms.models import get_model
from fms.models.gpt_bigcode import _gpt_bigcode_factory_factory, GPTBigCodeConfig
from fms.models.hf import to_hf_api
from fms.utils import generation, tokenizers, serialization
from fms.utils.generation import generate
from fms_extras.models import calico
from fms_extras.models.hf import register_fms_models
register_fms_models()

# This example script validates the LLaMA implementation by running inference on a couple of prompts.
#
# Example usage with single-GPU 7B model on slurm, with torch.compile and determinstic behavior:
# CUBLAS_WORKSPACE_CONFIG=:4096:8 srun -N 1 --gres=gpu:1 python scripts/inference.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --compile --deterministic
# Example usage of 13B model on 2 GPUs with Tensor Parallel:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 scripts/inference.py --model_path=~/models/13B-F --tokenizer=~/models/tokenizer.model --distributed

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
    default="7b",
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
    action="store_false",
    help="Disable the kv-cache (on by default)",
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
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)
print(device)

# torch.set_default_dtype(torch.half)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    torch.use_deterministic_algorithms(True)

if args.distributed:
    dist.init_process_group()
    # Fix until PT 2.3
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

print("loading model")
if args.distributed:
    distr_param = "tp"
else:
    if torch.cuda.device_count() > 1 and world_size == 1:
        distr_param = "mp"
    else:
        distr_param = None

def _hf_sd_to_fms_sd(hf_sd: Mapping) -> Mapping:
    import re

    replacements = [
        ("lm_head.weight", "head.weight"),
        (r"^transformer.wte.weight", "base_model.embedding.weight"),
        (r"^transformer.wpe.weight", "base_model.position_embedding.weight"),
        (r"^transformer.ln_f", "base_model.dec_norm"),
        (r"^transformer.h", "base_model.layers"),
        # need to do kqv manually
        (r"attn\.c_proj", "attn.dense"),
        (r"mlp\.c_fc", "ff_sub_layer.w1"),
        (r"mlp\.c_proj", "ff_sub_layer.w2"),
        (r"ln_1", "ln"),
        (r"ln_2", "ff_ln"),
    ]
    qkv_weight_pattern = re.compile("transformer.h.[0-9]+.attn.c_attn.weight")
    qkv_bias_pattern = re.compile("transformer.h.[0-9]+.attn.c_attn.bias")

    new_sd = {}
    for name, param in hf_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)

        new_sd[new_name] = param

        # qkv fused
        if bool(qkv_weight_pattern.match(name)):
            new_sd.pop(new_name)

            attn_splits = [
                3072,
                96,
                96
            ]

            prefix = new_name.replace("c_attn.weight", "")
            q, k, v = param.split(attn_splits, dim=0)

            new_sd[f"{prefix}query.weight"] = q
            new_sd[f"{prefix}key.weight"] = k
            new_sd[f"{prefix}value.weight"] = v
        elif bool(qkv_bias_pattern.match(name)):
            new_sd.pop(new_name)

            attn_splits = [
                3072,
                96,
                96
            ]

            prefix = new_name.replace("c_attn.bias", "")
            q, k, v = param.split(attn_splits, dim=0)

            new_sd[f"{prefix}query.bias"] = q
            new_sd[f"{prefix}key.bias"] = k
            new_sd[f"{prefix}value.bias"] = v

    return new_sd


serialization.register_adapter("gpt_bigcode", "hf.96", _hf_sd_to_fms_sd)

# class GPTBigCodeConfig(ModelConfig):
#     # This param default is based on https://huggingface.co/bigcode/gpt_bigcode-santacoder
#     src_vocab_size: int = 49157
#     # This param default is based on https://huggingface.co/bigcode/gpt_bigcode-santacoder
#     emb_dim: int = 2048
#     nheads: int = 12
#     nlayers: int = 12
#     pad_id: int = 0
#     max_expected_seq_len: int = 512
#     hidden_grow_factor: float = 4.0
#     activation_fn: str = "gelu-tanh"
#     p_dropout: float = 0.0
#     emb_dropout: float = 0.0
#     multiquery_attn: bool = True
#     ln_eps: float = 1e-5

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    device_type=args.device_type,
    source=args.model_source,
    distributed_strategy=distr_param,
    group=dist.group.WORLD,
)
tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
print("loading complete on rank", local_rank)

if args.compile:
    print("compiling model")
    # Bug with kv-cache in PT2.1
    torch._inductor.config.joint_graph_constant_folding = False
    # compiling can make first inference pass slow
    model = torch.compile(model, mode=args.compile_mode)


def ids_for_prompt(prompt):
    tokens = tokenizer.tokenize(prompt)
    # tokens = ["<s>"] + tokens
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

    # prompt1 = template.format(
    #     "Provide a list of instructions for preparing chicken soup."
    # )
    # prompt1 = template.format(
    #     "Add a boilerplate conda environment with numpy and pytorch"
    # )
    prompt1 = template.format(
        "Write me a function in python that performs a bubble sort"
    )
    prompt2 = template.format("Explain some popular greetings in Spanish.")

prompt1_str = prompt1
prompt1 = ids_for_prompt(prompt1)
prompt2 = ids_for_prompt(prompt2)

max_len = max([len(prompt) for prompt in [prompt1, prompt2]])
# prompt1 = pad_prompt(prompt1, max_len)
# LLaMA 7B did better on the spanish prompt vs 13B.
# TODO: add a better english prompt to demonstrate padding/batching.
# prompt2 = pad_prompt(prompt2, max_len)
# ids = torch.stack((prompt2, prompt1), dim=0)

ids = prompt1.unsqueeze(0)


def print_result(result):
    if local_rank != 0:
        return
    # stop at EOS token if present
    result = generation.truncate_after_eos(
        result, tokenizer.convert_tokens_to_ids("</s>")
    )
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
    # if model.config.ntk_scaling:
    #     max_seq_len = max(max_len, model.config.max_expected_seq_len)
    # else:
    # without ntk scaling, extending the seq length too far gives bogus results.
    max_seq_len = model.config.max_expected_seq_len

    print(next(model.parameters()).device)

    result = generate(
        model,
        ids,
        max_new_tokens=100,
        use_cache=use_cache,
        do_sample=do_sample,
        max_seq_len=max_seq_len,
    )
    for i in range(result.shape[0]):
        print_result(result[i])


print("generating output", local_rank)
do_sample = [False]
use_cache = [
    args.no_use_cache
]  # True/False are identical with greedy iff `torch.use_deterministic_algorithms(True)`
for sample, cache in itertools.product(do_sample, use_cache):
    infer(cache, sample)

# model = to_hf_api(model, bos_token_id=2, pad_token_id=2, eos_token_id=2)
# from transformers import pipeline, AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#
# gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
# print(gen(prompt1_str, max_new_tokens=100))