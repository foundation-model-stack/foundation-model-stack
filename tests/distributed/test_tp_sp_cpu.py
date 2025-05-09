import os
import torch
import torch.distributed as dist
import datetime
import numpy as np

from fms.models.llama import LLaMA, LLaMAConfig
from torch.distributed._tensor import distribute_tensor, Shard
from fms.distributed.strategy import TensorParallelStrategy
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    PrepareModuleOutput,
)

import torch.distributed._functional_collectives as collectives
from torch.distributed._functional_collectives import all_gather_tensor as _real_all_gather_tensor

# Hook to trace when collectives are triggered
def debug_all_gather_tensor(tensor, gather_dim: int = 0, group=None):
    import traceback
    print(f"\n[DEBUG] all_gather_tensor called on rank {dist.get_rank()}")
    print(f"        tensor shape: {tensor.shape}, gather_dim: {gather_dim}, group type: {type(group)}")
    print("        Call stack:")
    print("".join(traceback.format_stack(limit=4)))
    return _real_all_gather_tensor(tensor, gather_dim=gather_dim, group=group)


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

collectives.all_gather_tensor = debug_all_gather_tensor
collectives.reduce_scatter_tensor = debug_reduce_scatter_tensor


def setup_distributed(world_size=1, rank=0):
    world_size = int(os.environ.get("WORLD_SIZE", world_size))
    rank = int(os.environ.get("RANK", rank))

    print(f"[setup_distributed] Using: RANK={rank}, WORLD_SIZE={world_size}")
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    
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
    print(f"[Rank {rank}] Process group initialized")

def debug_tensor_parallel_strategy():
    setup_distributed()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    wandb_enabled = False
    
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

    config = LLaMAConfig(nlayers=2, fused_weights=False)
    print(f"[Rank {rank}] Creating model...")
    model = LLaMA(config=config, distributed_strategy=strategy)

    print(f"[Rank {rank}] Running forward pass...")

    # sequence_lengths = [5, 9, 7] # C1 + 2
    sequence_lengths = [1] # C3
    batch_size = len(sequence_lengths)
    max_seq_len = max(sequence_lengths)

    # C1: Multiple sequences with varying lengths
    # Pad all sequences in the batch to the longest sequence
    batch = []
    for seq_len in sequence_lengths:
        x = torch.randint(0, config.src_vocab_size, (seq_len,))
        pad_amount = max_seq_len - seq_len
        if pad_amount > 0:
            x = torch.nn.functional.pad(x, (0, pad_amount), value=0)
        batch.append(x)

    batch = torch.stack(batch)  # Shape: (batch_size, max_seq_len)

    # C2: If max_seq_len not divisible by world_size, pad to closest higher multiple
    padded_len = ((max_seq_len + world_size - 1) // world_size) * world_size

    # C3: If sequence length still < world_size, pad to world_size
    if padded_len < world_size:
        padded_len = world_size

    if padded_len > max_seq_len:
        pad_amount = padded_len - max_seq_len
        batch = torch.nn.functional.pad(batch, (0, pad_amount), value=0) 

    print(f"[Rank {rank}] Final batch shape: {batch.shape} (batch_size={batch_size}, sequence_length={padded_len})")

    with torch.no_grad():
        out = model(batch)

    print(f"[Rank {rank}] Output shape: {out.shape}")

    # Log results only form 1 process
    if rank == 0 and wandb_enabled:
        wandb.config.update({
            "output_shape": str(out.shape),
            "original_sequence_lengths": sequence_lengths,
            "padded_sequence_length": padded_len,
            "batch_size": batch_size,
            "success": True
        })
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    debug_tensor_parallel_strategy()