import argparse
import logging
import os
import random
import numpy as np
import torch
import time
from torch import distributed as dist
from fms import models
from fms.utils import fusion, tokenizers
import csv

SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096]

parser = argparse.ArgumentParser(description="Benchmark attention runtime for various sequence lengths.")
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument("--architecture", type=str, default="llama")
parser.add_argument("--variant", type=str, default="7b")
parser.add_argument("--tokenizer", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--deterministic", action="store_true", help="Set seeds and torch.use_deterministic_algorithms? Requires env variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`.")
parser.add_argument("--unfuse_weights", action="store_true", help="Unfuse any fused weight modules that support the unfuse_weights method.")
parser.add_argument("--paged", action="store_true", help="Enable paged attention via env var.")
parser.add_argument("--output_csv", type=str, default=None, help="Path to output CSV file.")
args = parser.parse_args()

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

if args.output_csv:
    output_csv = args.output_csv
else:
    output_csv = f"attention_runtime_{'paged' if args.paged else 'default'}.csv"

import csv
results = []
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["seq_len", "runtime_ms_with_cache", "runtime_ms_without_cache"])
    for seq_len in SEQUENCE_LENGTHS:
        print(f"[INFO] Benchmarking seq_len={seq_len} ...", flush=True)
        ids = torch.randint(tokenizer.vocab_size(), (BATCH_SIZE, seq_len), device=device, dtype=torch.long)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        ids = ids.detach()
        def run():
            with torch.no_grad():
                logits = model.forward(ids, use_cache=True)
                del logits
                torch.cuda.empty_cache()
        # Warmup
        try:
            for _ in range(2):
                run()
            torch.cuda.synchronize()
            start = time.time()
            run()
            torch.cuda.synchronize()
            end = time.time()
            runtime_ms_with_cache = (end - start) * 1000
            writer.writerow([seq_len, f"{runtime_ms_with_cache:.3f}", "N/A"])
            results.append((seq_len, runtime_ms_with_cache, None))
            print(f"[OK] seq_len={seq_len} runtime_ms_with_cache={runtime_ms_with_cache:.3f}", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e):
                writer.writerow([seq_len, "OOM", "N/A"])
                results.append((seq_len, None, None))
                print(f"[OOM] seq_len={seq_len}", flush=True)
                torch.cuda.empty_cache()
            else:
                raise

        # Without cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        ids = ids.detach()
        def run_without_cache():
            with torch.no_grad():
                logits = model.forward(ids, use_cache=False)
                del logits
                torch.cuda.empty_cache()
        # Warmup
        try:
            for _ in range(2):
                run_without_cache()
            torch.cuda.synchronize()
            start = time.time()
            run_without_cache()
            torch.cuda.synchronize()
            end = time.time()
            runtime_ms_without_cache = (end - start) * 1000
            writer.writerow([seq_len, "N/A", f"{runtime_ms_without_cache:.3f}"])
            results.append((seq_len, None, runtime_ms_without_cache))
            print(f"[OK] seq_len={seq_len} runtime_ms_without_cache={runtime_ms_without_cache:.3f}", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e):
                writer.writerow([seq_len, "N/A", "OOM"])
                results.append((seq_len, None, None))
                print(f"[OOM] seq_len={seq_len}", flush=True)
                torch.cuda.empty_cache()
            else:
                raise

print(f"Wrote results to {output_csv}")

# Plotting
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    def plot_both_curves(this_csv, is_paged):
        # Try to find the other csv
        base = 'paged' if is_paged else 'default'
        other = 'default' if is_paged else 'paged'
        this_label = f"paged={is_paged}"
        other_csv = this_csv.replace(base, other)
        plt.figure(figsize=(8,5))
        # Plot this run
        xs = [s for s, ms in results if ms is not None]
        ys = [ms for s, ms in results if ms is not None]
        plt.plot(xs, ys, marker='o', label=this_label)
        # Try to plot the other run if it exists
        if os.path.exists(other_csv):
            df = pd.read_csv(other_csv)
            xs2 = df['seq_len'].values
            ys2 = [float(x) if x != 'OOM' else None for x in df['runtime_ms_with_cache']]
            xs2 = [x for x, y in zip(xs2, ys2) if y is not None]
            ys2 = [y for y in ys2 if y is not None]
            plt.plot(xs2, ys2, marker='o', label=f"paged={'default' in other_csv}")
        else:
            print(f"[INFO] {other_csv} not found. Only plotting this run.")
        plt.xlabel("Sequence Length")
        plt.ylabel("Runtime (ms)")
        plt.title("Attention Runtime vs Sequence Length")
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
    plot_both_curves(output_csv, args.paged)
except ImportError:
    print(f"[INFO] matplotlib or pandas not installed. Plot manually from {output_csv}")