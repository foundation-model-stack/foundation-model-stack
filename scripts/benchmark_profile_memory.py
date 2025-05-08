import os
import torch
import argparse
from fms import models
from fms.utils import tokenizers

SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SIZE = 1

parser = argparse.ArgumentParser(description="Profile peak memory usage for various sequence lengths.")
parser.add_argument("--architecture", type=str, default="llama")
parser.add_argument("--variant", type=str, default="7b")
parser.add_argument("--tokenizer", type=str, required=True)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument("--paged", action="store_true", help="Enable paged attention via env var.")
args = parser.parse_args()

if args.paged:
    os.environ["FMS_ATTENTION_ALGO"] = "paged"
else:
    os.environ.pop("FMS_ATTENTION_ALGO", None)

device = torch.device(args.device_type)
model = models.get_model(args.architecture, args.variant, device_type=args.device_type)
model.eval()
tokenizer = tokenizers.get_tokenizer(args.tokenizer)

print(f"# Peak Memory Profile (paged={args.paged})")
print("# seq_len\tpeak_memory_gb")

for seq_len in SEQUENCE_LENGTHS:
    ids = torch.randint(tokenizer.vocab_size(), (BATCH_SIZE, seq_len), device=device, dtype=torch.long)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model.forward(ids, use_cache=True)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
    print(f"{seq_len}\t{peak_mem:.4f}") 