import os
import torch
import argparse
import threading
import timeit
import wandb
import csv
import datetime
import matplotlib.pyplot as plt
from fms import models
from fms.utils import tokenizers

SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SIZE = 1
NUM_REQUESTS = 4

parser = argparse.ArgumentParser(description="Profile throughput for various sequence lengths.")
parser.add_argument("--architecture", type=str, default="llama")
parser.add_argument("--variant", type=str, default="7b")
parser.add_argument("--tokenizer", type=str, required=True)
parser.add_argument("--device_type", type=str, default="cuda")
parser.add_argument("--paged", action="store_true", help="Enable paged attention via env var.")
parser.add_argument("--output_csv", type=str, help="Filename (without timestamp) for CSV results")
args = parser.parse_args()

# -----------------------------------------------------------
# Output directory and timestamped filenames
# -----------------------------------------------------------
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "final_project")
os.makedirs(OUT_DIR, exist_ok=True)

if args.output_csv:
    base_csv = os.path.splitext(os.path.basename(args.output_csv))[0]
    output_csv = os.path.join(OUT_DIR, f"{base_csv}_{TIMESTAMP}.csv")
else:
    output_csv = os.path.join(
        OUT_DIR, f"throughput_profile_{'paged' if args.paged else 'default'}_{TIMESTAMP}.csv"
    )
plot_path = output_csv.replace(".csv", "_plot.png")

# ────────────────────────────────────────────────────────────
# Initialize Weights & Biases run so every benchmark is logged
# ────────────────────────────────────────────────────────────
attention_algo = "paged" if args.paged else "default"
wandb_run = wandb.init(
    project="hpml-final-project",
    entity="nsd2147-columbia-university",
    name=f"{args.architecture}-{args.variant}-{attention_algo}-throughput",
    tags=[attention_algo, "throughput_profile"],
    config=vars(args),
)

if args.paged:
    os.environ["FMS_ATTENTION_ALGO"] = "paged"
else:
    os.environ.pop("FMS_ATTENTION_ALGO", None)

device = torch.device(args.device_type)
model = models.get_model(args.architecture, args.variant, device_type=args.device_type)
model.eval()
tokenizer = tokenizers.get_tokenizer(args.tokenizer)

print(f"# Throughput Profile (paged={args.paged})")
print("# seq_len\tthroughput_tokens_per_sec\tavg_latency_ms_per_token\tpeak_memory_gb")
results = []
csv_file = open(output_csv, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["seq_len", "throughput_tokens_per_sec", "avg_latency_ms_per_token", "peak_memory_gb"])

for seq_len in SEQUENCE_LENGTHS:
    requests = [
        torch.randint(tokenizer.vocab_size(), (BATCH_SIZE, seq_len), device=device, dtype=torch.long)
        for _ in range(NUM_REQUESTS)
    ]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    results = [None] * NUM_REQUESTS
    def run_forward(i):
        with torch.no_grad():
            results[i] = model.forward(requests[i], use_cache=True)
    threads = [threading.Thread(target=run_forward, args=(i,)) for i in range(NUM_REQUESTS)]
    start_time = timeit.default_timer()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    total_tokens = NUM_REQUESTS * BATCH_SIZE * seq_len
    throughput = total_tokens / total_time if total_time > 0 else float('inf')
    avg_latency = (total_time / total_tokens) * 1000 if total_tokens > 0 else 0  # ms per token
    peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB
    # Log to Weights & Biases
    wandb.log({
        "seq_len": seq_len,
        "throughput_tokens_per_sec": throughput,
        "avg_latency_ms_per_token": avg_latency,
        "peak_memory_gb": peak_mem
    })
    print(f"{seq_len}\t{throughput:.2f}\t{avg_latency:.2f}\t{peak_mem:.4f}") 
    writer.writerow([seq_len, f"{throughput:.2f}", f"{avg_latency:.2f}", f"{peak_mem:.4f}"])
    results.append((seq_len, throughput, avg_latency, peak_mem))

# After the loop: close csv, print path, and plot
csv_file.close()
print(f"[CSV] Wrote results to {output_csv}")

# Quick plot: throughput vs seq_len
if results:
    xs, ys, _, _ = zip(*results)
    plt.figure(figsize=(8,5))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Sequence Length")
    plt.ylabel("Throughput (tokens / sec)")
    plt.xscale("log", base=2)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.title("Throughput vs Sequence Length")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[PLOT] Saved plot to {plot_path}")

# Close the W&B run so it uploads all metadata.
wandb_run.finish()