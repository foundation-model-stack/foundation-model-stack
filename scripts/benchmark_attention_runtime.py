import os
import time
import torch
import argparse
from fms import models
from fms.utils import tokenizers
import numpy as np

SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SIZE = 1

parser = argparse.ArgumentParser(description="Benchmark attention runtime for various sequence lengths.")
parser.add_argument("--architecture", type=str, default="llama")
parser.add_argument("--variant", type=str, default="7b")
parser.add_argument("--tokenizer", type=str, required=True)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument("--paged", action="store_true", help="Enable paged attention via env var.")
args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.half)

if args.paged:
    os.environ["FMS_ATTENTION_ALGO"] = "paged"
else:
    os.environ.pop("FMS_ATTENTION_ALGO", None)

print("loading model")
model = models.get_model(args.architecture, args.variant, device_type=args.device_type)
model.eval()
tokenizer = tokenizers.get_tokenizer(args.tokenizer)
torch.set_grad_enabled(False)
print(f"loading complete on rank {local_rank}")

print(f"# Attention Runtime Benchmark (paged={args.paged})")
print("# seq_len\truntime_ms")

for seq_len in SEQUENCE_LENGTHS:
    input_ids = torch.randint(tokenizer.vocab_size(), (BATCH_SIZE, seq_len), device=device, dtype=torch.long)
    input_ids.requires_grad = False
    # Forward pass timing only
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    input_ids = input_ids.detach()
    input_ids.requires_grad = False
    def run():
        logits = model(input_ids)
        _ = logits.sum().item()  # Ensure computation
    # Warmup
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    start = time.time()
    run()
    torch.cuda.synchronize()
    end = time.time()
    runtime_ms = (end - start) * 1000
    print(f"{seq_len}\t{runtime_ms:.3f}") 