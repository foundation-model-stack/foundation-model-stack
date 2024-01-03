import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pyarrow as pa
from typing import Callable, Union

from fms.models import llama
from fms.modules.speculator import Speculator
from fms.utils.cache.paged import PagedKVCacheManager
from fms.utils.generation import speculative_generate
from transformers import LlamaForCausalLM

parser = argparse.ArgumentParser(description="Script to run inference on a LLaMA model")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the directory containing HF LLaMa weights",
)
parser.add_argument(
    "--speculator_path",
    type=str,
    required=True,
    help="Path to the checkpoint containing speculator weights (single .pth file, not HF weights)",
)
parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Path to the set of prompts (single .pth file)",
)
parser.add_argument(
    "--bsize", type=int, default=1, help="Number of sequences per batch"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="",
    help="Path to save the final scores, if needed",
)
args = parser.parse_args()


print("Job start!")

model = LlamaForCausalLM.from_pretrained(args.model_path)
model = llama.convert_hf_llama(model)
model.eval()
model.cuda()
model.to(dtype=torch.bfloat16)
model.rot_emb.compute_freqs_cis(model.shared.emb.weight.device, 4096)

print("Model loaded!")

print("initiating cache")
kv_cache = PagedKVCacheManager(
    model.config.nlayers,
    model.config.nheads,
    model.config.emb_dim,
    total_num_gpu_blocks=3818,
    dtype=model.shared.emb.weight.dtype,
)

data = torch.load(args.data_path)
print("Data prepared!")

test = Speculator(n_heads=3, emb_dim=5120)
test.load_state_dict(
    torch.load(
        args.speculator_path,
        map_location="cpu",
    )["model_state"]
)
test.cuda()
test.to(dtype=torch.bfloat16)

print("Speculator ready!")

torch.cuda.empty_cache()
bsize = args.bsize
steps = {}
outs = []
for k in [2, 5, 10, 25]:
    steps[k] = []
    for j in range(test // bsize + 1):
        seqs = data[j * bsize : j * bsize + bsize]
        inp = [torch.IntTensor(line).cuda() for line in seqs]
        with torch.no_grad():
            out, nsteps = speculative_generate(
                model,
                inp,
                test,
                new_tokens=100,
                max_seq_len=4096,
                top_k=k,
                kv_cache_manager=kv_cache,
            )
        if k == 5:
            outs += [line.squeeze().tolist() for line in out]
        for i in range(bsize):
            print(f"Ex {j*bsize+i}, topk={k}: {len(out[i])} tokens in {nsteps} steps.")

        adjusted_steps = [nsteps * 100 / len(line) for line in out]
        steps[k] += adjusted_steps

if len(args.output_path) > 0:
    torch.save(steps, os.path.join(args.output_path, "steps_for_100_at_k.pth"))
