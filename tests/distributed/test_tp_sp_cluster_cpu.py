import os
import torch
import torch.distributed as dist
import datetime
import time
import wandb
import matplotlib.pyplot as plt
import platform
import psutil

from fms.models.llama import LLaMA, LLaMAConfig
from fms.distributed.strategy import TensorParallelStrategy


# Initialize PyTorch distributed (Gloo backend) and W&B tracking
def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29501')
    os.environ["WANDB_API_KEY"] = "1b80de13ae8a9d78812516c11ee97a4c83010793"

    # Only rank 0 should log to Weights & Biases
    if rank == 0:
        wandb.init(
            project="fms-tp-sp",
            config={
                "model": "LLaMA",
                "nlayers": 2,
                "strategy": "TP vs TP+SP (CPU)",
                "vocab_size": 32000,
                "sequence_length": "256-1024",
                "device": "cpu"
            },
            reinit="finish_previous"
        )

    # Use FileStore for rendezvous between CPU-only ranks
    if not dist.is_initialized():
        store = dist.FileStore("/tmp/shared_file_store", world_size)
        dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            store=store,
            timeout=datetime.timedelta(seconds=60)
        )


# Print CPU hardware info from the rank 0 process
def log_cpu_info(rank):
    if rank == 0:
        print(f"[Rank {rank}] Running on host: {platform.node()}")
        print(f"[Rank {rank}] CPU Info:")
        print(f"  Architecture: {platform.machine()}")
        print(f"  Processor: {platform.processor()}")
        print(f"  Cores (physical/logical): {psutil.cpu_count(logical=False)}/{psutil.cpu_count(logical=True)}")
        print(f"  RAM: {round(psutil.virtual_memory().total / 1e9, 2)} GB")


# Benchmark one full run of either TP or TP+SP on CPU for 3 sequence lengths
def run_benchmark(sequence_parallel=False):
    latency = []
    memory_used = []

    os.environ["USE_SEQUENCE_PARALLELISM"] = str(sequence_parallel)
    setup_distributed()

    rank = dist.get_rank()
    device = torch.device("cpu")
    log_cpu_info(rank)

    strategy = TensorParallelStrategy()
    config = LLaMAConfig(nlayers=2, max_expected_seq_len=1024, fused_weights=False)
    model = LLaMA(config=config, distributed_strategy=strategy).to(device)
    model.eval()

    for seq_len in [256, 512, 1024]:
        x = torch.randint(0, config.src_vocab_size, (2, seq_len), device=device)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start = time.time()
        with torch.no_grad():
            _ = model(x)
        end = time.time()

        print(f"[Rank {rank}] [{'SP' if sequence_parallel else 'TP'}] SeqLen {seq_len}: {end - start:.2f}s")
        latency.append(end - start)
        memory_used.append(psutil.Process().memory_info().rss / 1e9)

    # Ensure all ranks complete before cleanup
    dist.barrier()
    dist.destroy_process_group()
    time.sleep(2)

    if rank == 0:
        wandb.finish()

    return latency, memory_used


if __name__ == "__main__":
    # First run: Sequence Parallel (SP) mode
    sp_times, sp_mem = run_benchmark(sequence_parallel=True)

    # Pause briefly between runs to avoid socket reuse race
    time.sleep(2)

    # Second run: Tensor Parallel (TP) only
    tp_times, tp_mem = run_benchmark(sequence_parallel=False)

    # Save plots to local output directory
    seq_lengths = [256, 512, 1024]
    output_dir = "distributed_tests_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot execution time comparison
    plt.figure(figsize=(10, 5))
    plt.plot(seq_lengths, sp_times, marker='o', label='Sequence Parallel (CPU)')
    plt.plot(seq_lengths, tp_times, marker='o', label='Tensor Parallel (CPU)')
    plt.title('Execution Time vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'execution_time_vs_seq_length_cpu.png'))
    plt.close()

    # Plot memory usage comparison
    plt.figure(figsize=(10, 5))
    plt.plot(seq_lengths, sp_mem, marker='o', label='Sequence Parallel (CPU)')
    plt.plot(seq_lengths, tp_mem, marker='o', label='Tensor Parallel (CPU)')
    plt.title('Memory Usage vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Usage (GB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'memory_usage_vs_seq_length_cpu.png'))
    plt.close()
