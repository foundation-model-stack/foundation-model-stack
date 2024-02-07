import argparse
import functools
import itertools
import os
import statistics
import time
import timeit
from typing import Union, Callable, MutableMapping, Any

import torch
import torch._inductor.config
from torch._C._profiler import ProfilerActivity

from fms.modules.positions import compute_position_ids
from fms.utils import generation, tokenizers, print0
from text_generation_server.models import get_model


# This example script validates the LLaMA implementation by running inference on a couple of prompts.
#
# Example usage with single-GPU 7B model on slurm, with torch.compile and determinstic behavior:
# CUBLAS_WORKSPACE_CONFIG=:4096:8 srun -N 1 --gres=gpu:1 python scripts/inference.py --model_path=~/models/7B-F/ --tokenizer=~/models/tokenizer.model --compile --deterministic
# Example usage of 13B model on 2 GPUs with Tensor Parallel:
# srun -N 1 --gres=gpu:2 torchrun --nproc_per_node=2 scripts/inference.py --model_path=~/models/13B-F --tokenizer=~/models/tokenizer.model --distributed

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))

device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

torch.set_default_device(device)
torch.set_default_dtype(torch.half)

print("loading model")
model = get_model(
    model_name="/net/storage149/mnt/md0/jmrosenk/llama_weights/hf/7B-F",
    revision=None,
    deployment_framework="hf_custom_tp",
    dtype_str="float16",
    quantize=None,
    max_sequence_length=2048
)
model = model.model

tokenizer = tokenizers.get_tokenizer("/net/storage149/mnt/md0/jmrosenk/llama_weights/tokenizer.model")
model.eval()
torch.set_grad_enabled(False)
print("loading complete on rank", local_rank)

def trace_handler(p, output_path, extra_name=""):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace(f"{output_path}/trace_step{str(p.step_num)}_{extra_name}.json")

def single_prompt_generate(
    model: Union[Callable, torch.nn.Module],
    input_ids: torch.Tensor,
    max_seq_len: int = 2048,
    max_new_tokens: int = 256,
    print_single_token_time: bool = False
):
    batched = False
    if type(input_ids) == torch.Tensor:
        if input_ids.dim() != 1:
            batched = True
    else:
        raise RuntimeError("generate() requires a tensor of token ids as the prefix")

    if not batched:
        input_ids = input_ids.unsqueeze(0)

    result = input_ids
    next_input = input_ids
    kwargs: MutableMapping[str, Any] = dict()
    kwargs["past_key_values"] = None
    kwargs['cu_seqlens_q'] = None
    context_lengths = None

    if print_single_token_time:
        total_time = 0
    # comment out if you do not want to profile
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                skip_first=5,
                wait=0,
                warmup=3,
                active=1,
                repeat=1,
            ),
            on_trace_ready=functools.partial(trace_handler, output_path="/net/storage149/mnt/md0/jmrosenk/trace_generate_tgis", extra_name="0"),
            with_stack=True,
            profile_memory=True,
            record_shapes=True,
    ) as prof:
        for i in range(max_new_tokens):
            input_ids = next_input[:, -max_seq_len:]

            # get the cache data and position ids if using cache
            if i == 0:
                num_tokens_per_sequence = torch.count_nonzero(
                    input_ids.T, dim=0
                ).tolist()

                kwargs['cu_seqlens'] = torch.tensor([0] + num_tokens_per_sequence, dtype=torch.int32, device="cuda")
                kwargs['pre_allocate_past_size'] = num_tokens_per_sequence[0] + max_new_tokens
            else:
                if kwargs['cu_seqlens_q'] is None:
                    kwargs['cu_seqlens_q'] = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

                context_lengths[0] += 1
                kwargs['cu_seqlens'] = kwargs['cu_seqlens'] + kwargs['cu_seqlens_q']
                num_tokens_per_sequence = [1 for _ in range(input_ids.size(0))]
                kwargs['pre_allocate_past_size'] = None

            position_ids = compute_position_ids(num_tokens_per_sequence, context_lengths)
            if context_lengths is None:
                context_lengths = num_tokens_per_sequence
            kwargs["max_s"] = context_lengths[0]
            kwargs["position_ids"] = torch.tensor(position_ids, dtype=torch.int32, device="cuda").squeeze(0)

            input_ids = input_ids.squeeze(0)
            if i != 0 and print_single_token_time:
                start = time.time()

            output = model(input_ids, **kwargs)

            if i != 0 and print_single_token_time:
                end = time.time()
                total_time += (end - start)


            logits, past_key_values = output
            logits = logits[-1:, :]

            next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()
            result = torch.cat((result, next_val), dim=-1)


            next_input = next_val
            kwargs["past_key_values"] = past_key_values
            # comment out if you do not want to profile
            prof.step()

    if not batched:
        result = result[0]

    if print_single_token_time:
        print(total_time / (max_new_tokens-1))

    return result


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
    pads = torch.tensor(pad_ids, device=device)
    return torch.cat((pads, prompt))



template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

prompt1 = template.format(
    "Provide a list of instructions for preparing chicken soup."
)

prompt1 = ids_for_prompt(prompt1)
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


def infer():
    # With greedy generation (do_sample=False) we _should_ always get the same results.
    # There is currently a bug in start_pos for batched rotary embeddings that can lead
    # varying results for the same prompt.
    if local_rank == 0:
        print("printing output from prompt as test")
        print("==================")
    max_seq_len = 2048

    result = single_prompt_generate(
        model,
        ids,
        max_new_tokens=100,
        max_seq_len=max_seq_len,
    )
    for i in range(result.shape[0]):
        print_result(result[i])

print("generating output", local_rank)
# simple print to check, also good warmup
infer()


SEQ_LEN = 500
ids = torch.randint(
    tokenizer.vocab_size(), (1, SEQ_LEN), device=device, dtype=torch.long
)
MAX_NEW_TOKENS = 50
repeat = 5

def end_to_end(model, print_single_token_time):
    result = single_prompt_generate(
        model,
        ids,
        max_new_tokens=MAX_NEW_TOKENS,
        print_single_token_time=print_single_token_time,
    )
    if local_rank == 0:
        assert (
            result.size()[-1] == SEQ_LEN + MAX_NEW_TOKENS
        ), f"{result.size()}, {SEQ_LEN}, {MAX_NEW_TOKENS}"

    torch.cuda.synchronize()
    return result

def log_result(result):
    if local_rank == 0:
        median = statistics.median(result)
        per_token = median / MAX_NEW_TOKENS
        ms = per_token * 1000
        print(f"\t{ms:0.2f} ms per token")

def bench_end_to_end(print_single_token_time):
    print_results = "forward pass isolated" if print_single_token_time else "full generate time"
    print0(f"printing results avg time per token: {print_results}")
    result = timeit.repeat(
        lambda: end_to_end(model, print_single_token_time),
        number=1,
        repeat=repeat,
    )
    if not print_single_token_time:
        log_result(result)

# bench_end_to_end(print_single_token_time=False)
# bench_end_to_end(print_single_token_time=True)
