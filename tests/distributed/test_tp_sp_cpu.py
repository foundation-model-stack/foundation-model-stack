import os
import torch
import torch.distributed as dist
import datetime
import numpy as np

from fms.models.llama import LLaMA, LLaMAConfig
from torch.distributed._tensor import distribute_tensor, Shard
from fms.distributed.strategy import TensorParallelStrategy

import torch.distributed._functional_collectives as collectives
from torch.distributed._functional_collectives import all_gather_tensor as _real_all_gather_tensor

import torch.distributed.distributed_c10d as c10d

# Hook to trace all_gather_tensor usage across ranks
def debug_all_gather_tensor(tensor, gather_dim: int = 0, group=None):
    import traceback
    print(f"\n[DEBUG] all_gather_tensor called on rank {dist.get_rank()}")
    print(f"        tensor shape: {tensor.shape}, gather_dim: {gather_dim}, group type: {type(group)}")
    print("        Call stack:")
    print("".join(traceback.format_stack(limit=4)))
    return _real_all_gather_tensor(tensor, gather_dim=gather_dim, group=group)

# Hook to trace reduce_scatter_tensor usage across ranks
_real_reduce_scatter_tensor = collectives.reduce_scatter_tensor
def debug_reduce_scatter_tensor(*args, **kwargs):
    import traceback
    tensor = args[0] if len(args) > 0 else kwargs.get("tensor", None)
    scatter_dim = kwargs.get("scatter_dim", None)
    group = kwargs.get("group", None)

    print(f"\n[DEBUG] reduce_scatter_tensor called on rank {dist.get_rank()}")
    print(f"        tensor shape: {tensor.shape if tensor is not None else 'Unknown'}")
    print(f"        scatter_dim: {scatter_dim}, group type: {type(group)}")
    print("        Call stack:")
    print("".join(traceback.format_stack(limit=4)))

    return _real_reduce_scatter_tensor(*args, **kwargs)

# Replace collectives with debug versions
collectives.all_gather_tensor = debug_all_gather_tensor
collectives.reduce_scatter_tensor = debug_reduce_scatter_tensor


def setup_distributed():
    # Initialize the default distributed process group (called once per process)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo" if not torch.cuda.is_available() else "nccl",
            timeout=datetime.timedelta(seconds=30)
        )

    # Register the group explicitly under the name "default"
    c10d._register_process_group("default", dist.group.WORLD)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}] Process group initialized with world size {world_size}")


def debug_tensor_parallel_strategy():
    setup_distributed()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    wandb_enabled = False

    # Initialize Weights & Biases logging on rank 0
    if rank == 0:
        try:
            import wandb
            wandb.init(project="fms-tp-sp")
            wandb_enabled = True
            print("[Rank 0] WandB initialized.")
        except Exception as e:
            print(f"[Rank 0] WandB initialization failed: {e}")

    print("[All Ranks] Creating TensorParallelStrategy...")
    strategy = TensorParallelStrategy()

    # Create small 2 layer LLaMA model with the given strategy
    config = LLaMAConfig(nlayers=2, fused_weights=False)
    print(f"[Rank {rank}] Creating model...")
    model = LLaMA(config=config, distributed_strategy=strategy)

    print(f"[Rank {rank}] Running forward pass...")

    # Simulate a batch with short sequence length(s)
    sequence_lengths = [1]  # Change to e.g., [5, 9, 7] for variable length testing
    batch_size = len(sequence_lengths)
    max_seq_len = max(sequence_lengths)

    # Pad each sequence in batch to max_seq_len
    batch = []
    for seq_len in sequence_lengths:
        x = torch.randint(0, config.src_vocab_size, (seq_len,))
        pad_amount = max_seq_len - seq_len
        if pad_amount > 0:
            x = torch.nn.functional.pad(x, (0, pad_amount), value=0)
        batch.append(x)
    batch = torch.stack(batch)

    # Ensure sequence length is divisible by world_size
    padded_len = ((max_seq_len + world_size - 1) // world_size) * world_size

    # If still smaller than world_size, pad further
    if padded_len < world_size:
        padded_len = world_size
    if padded_len > max_seq_len:
        pad_amount = padded_len - max_seq_len
        batch = torch.nn.functional.pad(batch, (0, pad_amount), value=0)

    print(f"[Rank {rank}] Final batch shape: {batch.shape} (batch_size={batch_size}, sequence_length={padded_len})")

    # Forward pass (no grad since it's for debugging mainly)
    with torch.no_grad():
        out = model(batch)

    print(f"[Rank {rank}] Output shape: {out.shape}")

    # Log metadata to WandB
    if rank == 0 and wandb_enabled:
        wandb.config.update({
            "output_shape": str(out.shape),
            "original_sequence_lengths": sequence_lengths,
            "padded_sequence_length": padded_len,
            "batch_size": batch_size,
            "success": True
        })
        wandb.finish()

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    debug_tensor_parallel_strategy()
