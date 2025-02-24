import argparse
import os
import time

import torch
import transformers
from torch import nn

from fms.models import llama
from fms.models.hf.llama import modeling_llama_hf
from fms.models.hf.utils import register_fms_models


# This script demonstrates how to use FMS model implementations with HF formatted
# weights.
#
# Requires first installing transformers, sentencepiece, protobuf, and torch >= 2.1.0

parser = argparse.ArgumentParser(
    description="Example script to load HF weights into an FMS model, and use them in the HF ecosystem"
)

parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="The path to a directory containing hugging-face formatted LLaMA weights and tokenizer",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="""q: how are you? a: I am good. How about you? q: What is the weather like today? a:""",
    help="An input prompt to seed output from the LM",
)

args = parser.parse_args()

local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device("cuda", local_rank)

tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path)
prompt = args.prompt
prompt_tokens = tokenizer.tokenize(prompt)
input_ids = tokenizer.convert_tokens_to_ids(prompt_tokens)
input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

# Create an instance of the huggingface model using huggingface weights
model = transformers.LlamaForCausalLM.from_pretrained(
    args.model_path, use_safetensors=True
)
model.to(device)
model = model.to(torch.half)


# make sure we always call generate the same way when comparing implementations.
def generate(model: nn.Module, new_tokens=25):
    return model.generate(
        input_ids=input_ids, use_cache=True, max_new_tokens=new_tokens, do_sample=False
    )


def timed_generation(model: nn.Module, run: str):
    start = time.time()
    result = generate(model)[0]
    end = time.time()
    print(f"{run} took {end - start:,.2f} seconds")
    actual_new = result.shape[0] - input_ids.shape[1]
    print(f"\t- Generated {actual_new} new tokens")
    print(f"\t- {(end - start) * 1000 / actual_new:,.2f} ms per generated token")
    print(f'\t- Generated text: "{tokenizer.decode(result[-actual_new:])}"')


# warmup
generate(model)
# HF implementation
timed_generation(model, run="Uncompiled HF implementation")

# Convert to an instance of the FMS implementation of LLaMA, which supports
# `torch.compile`
model = llama.convert_hf_llama(model)

# Adapt the FMS implementation back to the HF API, so it can be used in
# the huggingface ecosystem. Under the hood this is still the FMS
# implementation.
model = modeling_llama_hf.HFAdaptedLLaMAForCausalLM.from_fms_model(model)
model.to(device)
model = model.to(torch.half)
register_fms_models()

generate(model)
timed_generation(model, run="Uncompiled FMS model")

# Compile the underlying/wrapped model components.
# HF's generate doesn't actually call the original forward, so need to compile
# from HF adapted API.

# this seems to be needed in other inference scripts but lack of this line
# didn't fail here?
torch._inductor.config.joint_graph_constant_folding = False
model.decoder = torch.compile(model.decoder, dynamic=True)

# trying a few different variants of compile options...
# reduce-overhead will cause segfaults with non-contiguous kv-cache, HF
# generate API doesn't make them contiguous.
# model.decoder = torch.compile(model.decoder, mode='reduce-overhead', dynamic=True)

# First pass on a compiled model is often slower
timed_generation(model, run="Compiling FMS")
# Warmed-up
timed_generation(model, run="Compiled FMS model")
