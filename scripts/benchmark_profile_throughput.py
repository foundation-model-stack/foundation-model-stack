import os
import torch
import argparse
import threading
import timeit
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
args = parser.parse_args()

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
    print(f"{seq_len}\t{throughput:.2f}\t{avg_latency:.2f}\t{peak_mem:.4f}") 