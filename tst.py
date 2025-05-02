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

# Hook to trace if all_gather is triggered
# def debug_all_gather_tensor(tensor, gather_dim: int = 0, group=None):
#     import traceback
#     print(f"\n[DEBUG] all_gather_tensor called on rank {dist.get_rank()}")
#     print(f"        tensor shape: {tensor.shape}, gather_dim: {gather_dim}, group type: {type(group)}")
#     print("        Call stack:")
#     print("".join(traceback.format_stack(limit=4)))
#     return _real_all_gather_tensor(tensor, gather_dim=gather_dim, group=group)

# collectives.all_gather_tensor = debug_all_gather_tensor

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
    
    if rank == 0:
        try:
            import wandb
            wandb.init(project="fms-tp-sp", config={
                "model": "LLaMA",
                "nlayers": 2,
                "strategy": "TensorParallel",
                "vocab_size": 32000,
                "sequence_length": 16,
                "batch_size": 2,
            })
            wandb_enabled = True
            print("[Rank 0] WandB initialized.")
        except Exception as e:
            print(f"[Rank 0] WandB initialization failed: {e}")
            wandb_enabled = False
    else:
        wandb_enabled = False

    print("[All Ranks] Creating TensorParallelStrategy...")
    strategy = TensorParallelStrategy()

    config = LLaMAConfig(nlayers=2, fused_weights=False)
    print(f"[Rank {rank}] Creating model...")
    model = LLaMA(config=config, distributed_strategy=strategy)

    print(f"[Rank {rank}] Running forward pass...")
    x = torch.randint(0, config.src_vocab_size, (1, 8))
    # x = distribute_tensor(x, strategy.device_mesh, placements=[Shard(1)])

    with torch.no_grad():
        out = model(x)

    print(f"[Rank {rank}] Output shape: {out.shape}")

    if rank == 0 and wandb_enabled:
        output_np = out.cpu().numpy()
        if np.isnan(output_np).any():
            print("[Rank 0] Output contains NaNs — skipping histogram logging.")
            wandb.log({"success": False, "error": "Output contains NaNs"})
        else:
            wandb.log({
                "output_shape": str(out.shape),
                "output_hist": wandb.Histogram(output_np),
                "success": True
            })
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    debug_tensor_parallel_strategy()