import argparse
import itertools
import logging
import os
from re import M
import traceback

import torch
import torch._inductor.config
from torch import distributed as dist

from fms.models import get_model
from fms.utils import generation, tokenizers
from fms.utils.generation import generate

import gc
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
    "--fp8",
    action="store_true",
    help="Use float8_experimental Float8Linear to swap linear layers",
)
parser.add_argument(
    "--fp8_linear_type",
    type=str,
    default='dasw',
    choices=['dadw', 'dasw', 'sw', 'ns'],
    help="Choose the float8 linear type",
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
    choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
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
parser.add_argument('--max_new_tokens', type=int, default=2048)
parser.add_argument('--max_seq_len', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--reps', type=int, default=10)
args = parser.parse_args()
print(args)

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.bfloat16)

# requires setting environment variable: `CUBLAS_WORKSPACE_CONFIG=:4096:8`
if args.deterministic:
    torch.use_deterministic_algorithms(True)

if args.distributed:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29508'
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

# torch.cuda.memory._record_memory_history(max_entries=500000)

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

prefill_model = model
decode_model = model

if args.fp8:
    from float8_experimental.float8_linear import Float8Linear, Float8DASWLinear, Float8SWLinear
    from float8_experimental.float8_linear_utils import (
        swap_linear_with_float8_linear,
    )
    fp8LinearDict = {
        'dadw': Float8Linear,
        'dasw': Float8DASWLinear,
        'sw':   Float8SWLinear,
        'ns':   None,
    }
    print("casting the model weights to FP8")
    model = swap_linear_with_float8_linear(model, fp8LinearDict[args.fp8_linear_type])

if args.compile:
    print("compiling model")
    # compiling can make first inference pass slow
    # torch._inductor.config.allow_buffer_reuse = False
    torch._inductor.config.fx_graph_cache = True
    prefill_model = torch.compile(model, fullgraph=True)
    decode_model = torch.compile(model, mode=args.compile_mode, fullgraph=True)

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
# prompt1 = pad_prompt(prompt1, max_len)
# LLaMA 7B did better on the spanish prompt vs 13B.
# TODO: add a better english prompt to demonstrate padding/batching.
# prompt2 = pad_prompt(prompt2, max_len)
# ids = torch.stack((prompt2, prompt1), dim=0)

ids = prompt1.unsqueeze(0)
# ids = torch.randint(0, 32000, (1, 1024), device=device)
# ids = torch.randint(0, 32000, (256, 128), device=device)
# ids = torch.randint(0, 32000, (args.batch_size, args.max_seq_len), device=device)

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
    if model.config.ntk_scaling:
        max_seq_len = max(max_len, model.config.max_expected_seq_len)
    else:
        # without ntk scaling, extending the seq length too far gives bogus results.
        max_seq_len = model.config.max_expected_seq_len

    result = generate(
        prefill_model,
        decode_model,
        ids,
        max_new_tokens=args.max_new_tokens,
        use_cache=use_cache,
        do_sample=do_sample,
        max_seq_len=max_seq_len,
    )
    for i in range(result.shape[0]):
       print_result(result[i])

# input("Press Enter to continue...")
import time
print("generating output", local_rank)
do_sample = [False]
use_cache = [
    args.no_use_cache
]  # True/False are identical with greedy iff `torch.use_deterministic_algorithms(True)`
torch.cuda.profiler.start()
def trace_handler(p, output_path, extra_name=""):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
    print(output)
    p.export_chrome_trace(
        f"{output_path}/trace_step{str(p.step_num)}_{extra_name}.json"
    )
class PrintingMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **kwargs)
        if any(s in str(func) for s in ["mm", "matmul", "_to_copy", "attention", "silu", "fp8_fast"]):
            print(f"{func.__module__}.{func.__name__}({args}, {kwargs}) = {out}")
            if isinstance(out, tuple):
                debug_out = out[0]
            else:
                debug_out = out
            if isinstance(out, torch.Tensor):
                if torch.any(torch.isnan(out)):
                    print(f"This op produced {torch.sum(torch.isnan(out))} NaNs!!")
                    traceback.print_stack()
        return out
# with torch.profiler.profile(
#         activities=[
#             torch.profiler.ProfilerActivity.CPU,
#             torch.profiler.ProfilerActivity.CUDA,
#         ],
#         on_trace_ready=functools.partial(
#             trace_handler,
#             output_path="/net/storage149/mnt/md0/aviros/foundation-model-stack",
#             extra_name=str(local_rank),
#         ),
#         with_stack=True,
#         profile_memory=True,
#         record_shapes=True,
#         schedule=torch.profiler.schedule(skip_first=2, active=1, wait=0, warmup=1)
#     ) as prof:

try:
    with torch.no_grad():
        for sample, cache in itertools.product(do_sample, use_cache):
            #with torch.profiler.profile(
            #     activities=[
            #         torch.profiler.ProfilerActivity.CPU,
            #         torch.profiler.ProfilerActivity.CUDA,
            #     ],
            #     profile_memory=True,
            #     #schedule=torch.profiler.schedule(
            #     #        wait=1,
            #     #        warmup=1,
            #     #        active=1,
            #     #        repeat=1),
            #) as prof:
            # with PrintingMode():
                for _ in range(args.reps):
                #for _ in [1]:
                    # input("Press Enter to continue...")
                    torch.compiler.cudagraph_mark_step_begin()
                    t0 = time.time_ns()
                    infer(cache, sample)
                    torch.cuda.synchronize()
                    t1 = time.time_ns()
                    dt = float(t1 - t0) / 1E9
                    print("total inference time: ", dt)
                    torch.cuda.empty_cache()
                    # prof.step()
                    # input("Press Enter to continue...")
            #print(prof.key_averages().table(sort_by="self_cuda_memory_usage",row_limit=10))
            # prof.export_chrome_trace("trace.json")
except torch.cuda.OutOfMemoryError as e:
    print(e)
finally:
    # prof.step()
    torch.cuda.profiler.stop()

    # torch.cuda.memory._dump_snapshot("./fp8_memory.pickle")
