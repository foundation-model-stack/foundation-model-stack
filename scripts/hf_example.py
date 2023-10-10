import argparse
import transformers
from fms.models import llama
import torch
from fms.models.hf.llama import modeling_llama_hf
from transformers import pipeline
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
args = parser.parse_args()

# Create an instance of the huggingface model using huggingface weights
model = transformers.LlamaForCausalLM.from_pretrained(args.model_path)
# Convert to an instance of the FMS implementation of LLaMA, which supports
# `torch.compile`
model = llama.convert_hf_llama(model)
# compile the model
model = torch.compile(model)
# Adapt the FMS implementation back to the HF API, so it can be used in
# the huggingface ecosystem. Under the hood this is still the FMS
# implementation.
model = modeling_llama_hf.LLaMAHFForCausalLM.from_fms_model(model)
register_fms_models()

# Use the compiled model as-usual in the HF ecosystem
tokenizer = transformers.LlamaTokenizer.from_pretrained(args.model_path)
llama_generator = pipeline(
    task="text-generation", model=model, max_new_tokens=25, tokenizer=tokenizer
)
result = llama_generator(
    """q: how are you? a: I am good. How about you? q: What is the weather like today? a:"""
)
print(result)
