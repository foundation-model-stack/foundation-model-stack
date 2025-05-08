import os
import torch
import argparse
from fms import models
from fms.utils import tokenizers
import csv

SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SIZE = 1

parser = argparse.ArgumentParser(description="Profile peak memory usage for various sequence lengths.")
parser.add_argument("--architecture", type=str, default="llama")
parser.add_argument("--variant", type=str, default="7b")
parser.add_argument("--tokenizer", type=str, required=True)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument("--paged", action="store_true", help="Enable paged attention via env var.")
parser.add_argument("--output_csv", type=str, default=None, help="Path to output CSV file.")
args = parser.parse_args()

if args.paged:
    os.environ["FMS_ATTENTION_ALGO"] = "paged"
else:
    os.environ.pop("FMS_ATTENTION_ALGO", None)

device = torch.device(args.device_type)
model = models.get_model(args.architecture, args.variant, device_type=args.device_type)
model.eval()
tokenizer = tokenizers.get_tokenizer(args.tokenizer)

default_csv = f"profile_memory_{'paged' if args.paged else 'default'}.csv"
output_csv = args.output_csv if args.output_csv else default_csv

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
                _ = model.forward(ids, use_cache=True)
            peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
            print(f"[OK] seq_len={seq_len} peak_memory_gb={peak_mem:.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[OOM] seq_len={seq_len}")
                torch.cuda.empty_cache()
                peak_mem = "OOM"
            else:
                raise
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