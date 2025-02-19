import argparse
import itertools
import os
import random

import numpy as np
import torch
import torch._inductor.config
from torch import distributed as dist
from torch.nn.attention.flex_attention import create_block_mask, or_masks

from fms.models import get_model
from fms.utils import fusion, generation, tokenizers
from fms.utils.generation import generate, pad_input_ids


# This example script validates the LLaMA implementation by running inference on a couple of prompts.
#
# Example usage with single-GPU 7B model on slurm, with torch.compile and determinstic behavior:
# CUBLAS_WORKSPACE_CONFIG=:4096:8 srun -N 1 --gres=gpu:1 python scripts/inference.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --compile --deterministic
# Example usage of 13B model on 2 GPUs with Tensor Parallel:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 scripts/inference.py --model_path=~/models/13B-F --tokenizer=~/models/tokenizer.model --distributed

parser = argparse.ArgumentParser(
    description="Script to test Granite runtime logic on auto-regressive models"
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

model = get_model(
    args.architecture,
    args.variant,
    model_path=args.model_path,
    device_type=args.device_type,
    source=args.model_source,
    distributed_strategy=distr_param,
    group=dist.group.WORLD,
    fused_weights=not args.unfuse_weights,
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

system_prompt = "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: February 14, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>"

context_template = "{}"

assistant_prompt = "<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"

a_context = context_template.format(
    "Repeat after me: The sun rises from the East."
)
b_context = context_template.format(
    "Repeat after me: The sun rises from the West."
)

system_tokens = ids_for_prompt(system_prompt).unsqueeze(0)

a_tokens = ids_for_prompt(a_context).unsqueeze(0)
b_tokens = ids_for_prompt(b_context).unsqueeze(0)

assistant_tokens = ids_for_prompt(assistant_prompt).unsqueeze(0)

sap_tokens = ids_for_prompt(system_prompt + a_context + assistant_prompt).unsqueeze(0)
sab_tokens = ids_for_prompt(system_prompt + a_context + b_context).unsqueeze(0)
sabp_tokens = ids_for_prompt(system_prompt + a_context + b_context + assistant_prompt).unsqueeze(0)
sb_tokens = ids_for_prompt(system_prompt + b_context).unsqueeze(0)
sba_tokens = ids_for_prompt(system_prompt + b_context + a_context).unsqueeze(0)
sbp_tokens = ids_for_prompt(system_prompt + b_context + assistant_prompt).unsqueeze(0)
sbap_tokens = ids_for_prompt(system_prompt + b_context + a_context + assistant_prompt).unsqueeze(0)

def derope(model, kv_cache, position_ids):
    for i, layer in enumerate(kv_cache):
        kv_cache[i] = (model.base_model.rot_emb.adjusted_tensor(layer[0].transpose(1,2), position_ids, reverse=True).transpose(1,2), layer[1])

def rerope(model, kv_cache, position_ids):
    for i, layer in enumerate(kv_cache):
        kv_cache[i] = (model.base_model.rot_emb.adjusted_tensor(layer[0].transpose(1,2), position_ids, reverse=False).transpose(1,2), layer[1])

def generate_context_info(model, context_tokens):
    extra_kwargs = {
        "position_ids": torch.arange(0, context_tokens.size(1), device=context_tokens.device, dtype=torch.int64).unsqueeze(0).repeat(context_tokens.size(0), 1)
    }
    result, context_cache = generate(
        model,
        context_tokens,
        max_new_tokens=1,
        use_cache=True,
        do_sample=False,
        max_seq_len=131072,
        extra_kwargs=extra_kwargs,
    )

    # De-RoPE the k cache
    derope(model, context_cache, extra_kwargs["position_ids"])

    return context_cache

def print_result(result, padding_kwargs=None):
    if local_rank != 0:
        return
    if padding_kwargs is not None:
        result = generation.trim_prefix(result)

    result = generation.trim_prefix(result, tokenizer.bos_token_id)

    # stop at EOS token if present and remove padding
    # result = generation.truncate_after_eos(result, tokenizer.eos_token_id)

    output_str = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(result)
    )

    print(output_str)
    print()


def generate_with_prefix_context(model, context_list, prompt_tokens):
    # Mix and rerope all context caches, create attention mask for Flex Attention and position_ids

    # context_list is a list[KVCache | tuple[KVCache]]
    # each element in the list is added sequentially to the context, with elements in tuple being added in parallel
    # context_type can be sequential for A -> B -> P generation
    # or it can be parallel for (A + B) -> P generation
    consumed_context_length = 0
    
    final_context = [None for _ in range(model.config.nlayers)] if len(context_list) > 0 else None

    for context_step in context_list:
        if isinstance(context_step, tuple):
            context_type = "parallel"
        else:
            context_type = "sequential"

        if context_type == "parallel":
            max_context_step_length = 0
            for context in context_step:
                max_context_step_length = max(max_context_step_length, context[0][0].size(2))
            
            for context in context_step:
                position_ids = torch.arange(consumed_context_length + max_context_step_length - context[0][0].size(2), consumed_context_length + max_context_step_length, dtype=torch.int64, device=context[0][0].device).unsqueeze(0).repeat(context[0][0].size(0), 1)
                rerope(model, context, position_ids)
                # Mix kv cache and position_ids
                for i, layer in enumerate(final_context):
                    if layer is None:
                        final_context[i] = context[i]
                    else:
                        final_context[i] = (
                            torch.cat([final_context[i][0], context[i][0]], dim=2),
                            torch.cat([final_context[i][1], context[i][1]], dim=2),
                        )

            consumed_context_length += max_context_step_length
        else:
            context_step_length = context_step[0][0].size(2)
            position_ids = torch.arange(consumed_context_length, consumed_context_length + context_step_length, dtype=torch.int64, device=context_step[0][0].device).unsqueeze(0).repeat(context_step[0][0].size(0), 1)
            rerope(model, context_step, position_ids)
            # Mix kv cache and position_ids
            for i, layer in enumerate(final_context):
                if layer is None:
                    final_context[i] = context_step[i]
                else:
                    final_context[i] = (
                        torch.cat([final_context[i][0], context_step[i][0]], dim=2),
                        torch.cat([final_context[i][1], context_step[i][1]], dim=2),
                    )
            
            consumed_context_length += context_step_length

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx + consumed_context_length >= kv_idx
    
    def context_mask(b, h, q_idx, kv_idx):
        return kv_idx <= consumed_context_length
    
    prefix_lm_causal = or_masks(context_mask, causal_mask)

    final_position_ids = torch.arange(consumed_context_length, consumed_context_length + prompt_tokens.size(1), dtype=torch.int64, device="cuda").unsqueeze(0).repeat(prompt_tokens.size(0), 1)

    if final_context is not None:
        extra_kwargs = {
            "past_key_value_states": final_context,
            "position_ids": final_position_ids,
            "mask_mod": prefix_lm_causal
        }
    else:
        extra_kwargs = {
            "position_ids": final_position_ids,
        }

    result, final_context = generate(
        model,
        prompt_tokens,
        max_new_tokens=128,
        use_cache=True,
        do_sample=False,
        max_seq_len=131072,
        extra_kwargs=extra_kwargs,
        eos_token_id=tokenizer.eos_token_id,
    )

    if len(result.shape) == 1:
        result = result.unsqueeze(0)

    for i in range(result.shape[0]):
        print_result(result[i])


s_cache = generate_context_info(model, system_tokens)
a_cache = generate_context_info(model, a_tokens)
b_cache = generate_context_info(model, b_tokens)
sb_cache = generate_context_info(model, sb_tokens)
sab_cache = generate_context_info(model, sab_tokens)
sba_cache = generate_context_info(model, sba_tokens)

print("======== No context, Prompt: S + A + P")
generate_with_prefix_context(model, [], sap_tokens)
print("======== Context: S -> A, Prompt: P")
generate_with_prefix_context(model, [s_cache, a_cache], assistant_tokens)
print("======== No context, Prompt: S + B + P")
generate_with_prefix_context(model, [], sbp_tokens)
print("======== Context: S -> B, Prompt: P")
generate_with_prefix_context(model, [s_cache, b_cache], assistant_tokens)
print("======== Context: S + B, Prompt: P")
generate_with_prefix_context(model, [sb_cache], assistant_tokens)

print("======== Context: S -> Par(A, B), Prompt: P")
generate_with_prefix_context(model, [s_cache, (a_cache, b_cache)], assistant_tokens)
print("======== Context: S -> Par(B, A), Prompt: P")
generate_with_prefix_context(model, [s_cache, (b_cache, a_cache)], assistant_tokens)

print("======== Context: S -> A -> B, Prompt: P")
generate_with_prefix_context(model, [s_cache, a_cache, b_cache], assistant_tokens)
print("======== Context: S -> B -> A, Prompt: P")
generate_with_prefix_context(model, [s_cache, b_cache, a_cache], assistant_tokens)

print("======== Context: S + A + B, Prompt: P")
generate_with_prefix_context(model, [sab_cache], assistant_tokens)
print("======== Context: S + B + A, Prompt: P")
generate_with_prefix_context(model, [sba_cache], assistant_tokens)