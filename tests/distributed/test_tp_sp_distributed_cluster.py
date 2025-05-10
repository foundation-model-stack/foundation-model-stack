import os
import torch
import torch.distributed as dist
import datetime
import time
import wandb
import matplotlib.pyplot as plt

from fms.models.llama import LLaMA, LLaMAConfig
from fms.distributed.strategy import TensorParallelStrategy


# Sets up torch.distributed and initializes wandb logging
def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    print(f"[setup_distributed] RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")

    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29501')

    try:
        wandb.init(project="fms-tp-sp", config={
            "model": "LLaMA",
            "nlayers": 32,
            "strategy": "TensorParallel",
            "vocab_size": 32000,
            "sequence_length": 16,
            "batch_size": 2,
        })
        print(f"[Rank {rank}] WandB initialized")
    except Exception as e:
        print(f"[Rank {rank}] WandB initialization failed: {e}")

    # Initialize the process group
    if not dist.is_initialized():
        store = dist.FileStore("/tmp/shared_file_store", world_size)
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            store=store,
            timeout=datetime.timedelta(seconds=30)
        )

    print(f"[Rank {rank}] Distributed process group initialized")


# Runs benchmark for sequence parallelism (SP)
def run_sequence_parallel_benchmark():
    latency = []
    memory_allocated = []
    memory_reserved = []

    print("Benchmarking sequence parallelism")
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(f"[Rank {rank}] Running on GPU {local_rank}: {torch.cuda.get_device_name(device)}")

    strategy = TensorParallelStrategy()
    print(f"[Rank {rank}] Strategy initialized")

    config = LLaMAConfig(nlayers=32, max_expected_seq_len=1024, fused_weights=False)
    model = LLaMA(config=config, distributed_strategy=strategy).to(device)
    model.eval()
    print(f"[Rank {rank}] Model created")

    # C1: Initial test with padding logic for short sequences
    sequence_lengths = [1]
    batch_size = len(sequence_lengths)
    max_seq_len = max(sequence_lengths)
    batch = []
    for seq_len in sequence_lengths:
        x = torch.randint(0, config.src_vocab_size, (seq_len,), device=device)
        pad_amount = max_seq_len - seq_len
        if pad_amount > 0:
            x = torch.nn.functional.pad(x, (0, pad_amount), value=0)
        batch.append(x)
    batch = torch.stack(batch)

    # C2 + C3: Pad to multiple of world size or at least world size
    padded_len = ((max_seq_len + world_size - 1) // world_size) * world_size
    if padded_len < world_size:
        padded_len = world_size
    if padded_len > max_seq_len:
        pad_amount = padded_len - max_seq_len
        batch = torch.nn.functional.pad(batch, (0, pad_amount), value=0)

    print(f"[Rank {rank}] Final batch shape: {batch.shape}")

    # Forward pass on padded input
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model(batch)
    torch.cuda.synchronize()
    end = time.time()

    print(f"[Rank {rank}] Output shape: {out.shape}")
    print(f"[Rank {rank}] Forward pass time: {end - start:.2f} sec")
    print(f"[Rank {rank}] Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    print(f"[Rank {rank}] Device mesh: {strategy.device_mesh}")

    print("Passed baseline sequence parallelism cases")

    # C4: Main benchmark loop across longer sequence lengths
    seq_lengths = [256, 512, 1024]
    for seq_len in seq_lengths:
        print(f"Starting sequence length {seq_len} test")
        x = torch.randint(0, config.src_vocab_size, (2, seq_len), device=device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model(x)
        torch.cuda.synchronize()
        end = time.time()

        print(f"[Rank {rank}] Forward pass time: {end - start:.2f} sec")
        print(f"[Rank {rank}] Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        print(f"[Rank {rank}] Max memory reserved: {torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB")
        latency.append(end - start)
        memory_allocated.append(torch.cuda.max_memory_allocated(device) / 1e9)
        memory_reserved.append(torch.cuda.max_memory_reserved(device) / 1e9)

    dist.destroy_process_group()
    return latency, memory_allocated, memory_reserved


# Runs benchmark for tensor parallelism (TP) only (no SP)
def run_tensor_parallel_benchmark():
    latency = []
    memory_allocated = []
    memory_reserved = []

    print("Benchmarking tensor parallelism without sequence parallelism")
    os.environ["USE_SEQUENCE_PARALLELISM"] = "False"
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    print(f"[Rank {rank}] Running on GPU {local_rank}: {torch.cuda.get_device_name(device)}")

    strategy = TensorParallelStrategy()
    print(f"[Rank {rank}] Strategy initialized")

    config = LLaMAConfig(nlayers=32, max_expected_seq_len=1024, fused_weights=False)
    model = LLaMA(config=config, distributed_strategy=strategy).to(device)
    model.eval()
    print(f"[Rank {rank}] Model created")

    # Benchmark loop over standard sequence lengths
    seq_lengths = [256, 512, 1024]
    for seq_len in seq_lengths:
        print(f"Starting sequence length {seq_len} test")
        x = torch.randint(0, config.src_vocab_size, (2, seq_len), device=device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model(x)
        torch.cuda.synchronize()
        end = time.time()

        print(f"[Rank {rank}] Forward pass time: {end - start:.2f} sec")
        print(f"[Rank {rank}] Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
        print(f"[Rank {rank}] Max memory reserved: {torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB")
        latency.append(end - start)
        memory_allocated.append(torch.cuda.max_memory_allocated(device) / 1e9)
        memory_reserved.append(torch.cuda.max_memory_reserved(device) / 1e9)

    dist.destroy_process_group()
    return latency, memory_allocated, memory_reserved


# Entry point — runs both benchmarks and generates comparison plots
if __name__ == "__main__":
    sp_times, sp_mem_allocated, sp_mem_reserved = run_sequence_parallel_benchmark()
    tp_times, tp_mem_allocated, tp_mem_reserved = run_tensor_parallel_benchmark()

    seq_length = [256, 512, 1024]
    output_dir = "distributed_tests_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot: Execution time
    plt.figure(figsize=(10, 5))
    plt.plot(seq_length, sp_times, marker='o', label='Sequence Parallel')
    plt.plot(seq_length, tp_times, marker='o', label='Tensor Parallel')
    plt.title('Execution Time vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'execution_time_vs_seq_length.png'))
    plt.close()

    # Plot: Memory allocated
    plt.figure(figsize=(10, 5))
    plt.plot(seq_length, sp_mem_allocated, marker='o', label='Sequence Parallel')
    plt.plot(seq_length, tp_mem_allocated, marker='o', label='Tensor Parallel')
    plt.title('Memory Allocated vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Allocated (GB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'memory_allocated_vs_seq_length.png'))
    plt.close()

    # Plot: Memory reserved
    plt.figure(figsize=(10, 5))
    plt.plot(seq_length, sp_mem_reserved, marker='o', label='Sequence Parallel')
    plt.plot(seq_length, tp_mem_reserved, marker='o', label='Tensor Parallel')
    plt.title('Memory Reserved vs Sequence Length')
    plt.xlabel('Sequence Length')
    plt.ylabel('Memory Reserved (GB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'memory_reserved_vs_seq_length.png'))
    plt.close()
