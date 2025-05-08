import argparse
import logging
import os
import random
import numpy as np
import torch
import time
import datetime
from torch import distributed as dist
from fms import models
from fms.utils import fusion, tokenizers
import csv

import wandb

SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048]
BATCH_SIZE = 1

parser = argparse.ArgumentParser(description="Benchmark attention runtime for various sequence lengths.")
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument("--architecture", type=str, default="llama")
parser.add_argument("--variant", type=str, default="7b")
parser.add_argument("--tokenizer", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--deterministic", action="store_true", help="Set seeds and torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`.")
parser.add_argument("--unfuse_weights", action="store_true", help="Unfuse any fused weight modules that support the unfuse_weights method.")
parser.add_argument("--paged", action="store_true", help="Enable paged attention via env var.")
parser.add_argument("--use-cache", action="store_true", help="Enable paged attention via env var.")
parser.add_argument("--output_csv", type=str, default=None, help="Path to output CSV file.")
args = parser.parse_args()

# ────────────────────────────────────────────────────────────
# Initialize Weights & Biases run so every benchmark is logged
# ────────────────────────────────────────────────────────────
attention_algo = "paged" if args.paged else "default"
wandb_run = wandb.init(
    project="hpml-final-project",
    entity="nsd2147-columbia-university",
    name=f"{args.architecture}-{args.variant}-{attention_algo}-mem-profile",
    tags=[attention_algo, "mem_profile", f"use_cache_{args.use_cache}"],
    config=vars(args),
)

local_rank = int(os.getenv("LOCAL_RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
if args.device_type == "cuda":
    device = torch.device(args.device_type, local_rank)
    torch.cuda.set_device(device)
else:
    device = torch.device(args.device_type)

torch.set_default_dtype(torch.half)

if args.deterministic:
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

if world_size > 1:
    dist.init_process_group()
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)

if args.paged:
    os.environ["FMS_ATTENTION_ALGO"] = "paged"
else:
    os.environ.pop("FMS_ATTENTION_ALGO", None)

use_cache = args.use_cache

print("loading model")
model = models.get_model(args.architecture, args.variant, device_type=args.device_type)
if args.unfuse_weights:
    print("unfusing weights")
    from fms.utils import fusion
    model = fusion.apply_unfuse_weights(model)
tokenizer = tokenizers.get_tokenizer(args.tokenizer)
model.eval()
torch.set_grad_enabled(False)
print(f"loading complete on rank {local_rank}")

BATCH_SIZE = args.batch_size

# Set up output directory, timestamp, and CSV output path
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "final_project")
os.makedirs(OUT_DIR, exist_ok=True)

if args.output_csv:
    base_csv = args.output_csv
    # If the user passed a path, strip any existing directory & extension to replace them
    base_csv = os.path.splitext(os.path.basename(base_csv))[0]
    output_csv = os.path.join(OUT_DIR, f"{base_csv}_{TIMESTAMP}.csv")
else:
    output_csv = os.path.join(
        OUT_DIR, f"profile_memory_{'paged' if args.paged else 'default'}_{TIMESTAMP}.csv"
    )

results = []
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["seq_len", "peak_memory_gb"])
    for seq_len in SEQUENCE_LENGTHS:
        ids = torch.randint(tokenizer.vocab_size(), (BATCH_SIZE, seq_len), device=device, dtype=torch.long)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                _ = model.forward(ids, use_cache=use_cache)
            peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
            print(f"[OK] seq_len={seq_len} peak_memory_gb={peak_mem:.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[OOM] seq_len={seq_len}")
                torch.cuda.empty_cache()
                peak_mem = "OOM"
            else:
                raise
        # Log to W&B (use NaN for OOM so charts are continuous)
        wandb.log({
            "seq_len": seq_len,
            "peak_memory_gb": float(peak_mem) if isinstance(peak_mem, float) else float('nan')
        })
        writer.writerow([
            seq_len,
            f"{peak_mem:.4f}" if isinstance(peak_mem, float) else peak_mem
        ])
        results.append((seq_len, peak_mem))

print(f"Wrote results to {output_csv}")

# Plotting
try:
    import matplotlib.pyplot as plt
    xs = [s for s, mem in results if isinstance(mem, float)]
    ys = [mem for s, mem in results if isinstance(mem, float)]
    plt.figure(figsize=(8,5))
    plt.plot(xs, ys, marker='o', color='green', label="Peak Memory Usage (GB)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Peak Memory (GB)")
    plt.title("Peak Memory Usage vs Sequence Length")
    plt.xscale("log", base=2)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plot_path = output_csv.replace('.csv', '_plot.png')
    plt.savefig(plot_path)
    print(f"[PLOT] Saved plot to {plot_path}")
    plt.show()
except ImportError:
    print(f"[INFO] matplotlib not installed. Plot manually from {output_csv}") 

# Close the W&B run so it uploads all metadata.
wandb_run.finish()