import os
import torch
import torch.distributed as dist
import datetime
import numpy as np

from fms.models.llama import LLaMA, LLaMAConfig
from fms.distributed.strategy import NoOpStrategy, TensorParallelStrategy
import wandb

def setup_distributed(world_size=1, rank=0):
    world_size = int(os.environ.get("WORLD_SIZE", world_size))
    rank = int(os.environ.get("RANK", rank))

    print(f"[setup_distributed] Using: RANK={rank}, WORLD_SIZE={world_size}")
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    
    if not dist.is_initialized():
        # file-based store for inter-process communication
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
    
    #  make sure all processes are ready
    dist.barrier()

def debug_tensor_parallel_strategy():
    setup_distributed()
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        try:
            wandb.init(project="fms-tp-sp", config={
                "model": "LLaMA",
                "nlayers": 2,
                "strategy": "TensorParallel",
                "vocab_size": 32000,
                "sequence_length": 16,
                "batch_size": 2,
            })
            print(f"[Rank {rank}] WandB initialized")
        except Exception as e:
            print(f"[Rank {rank}] WandB initialization error: {e}")
    
    try:
        print("Creating TensorParallelStrategy...")
        strategy = TensorParallelStrategy()
        print("Strategy created.")
        
        config = LLaMAConfig(nlayers=2, fused_weights=False)
        
        print("\nCreating model with TP strategy...")
        model = LLaMA(config=config, distributed_strategy=strategy)
        print("Model created with 2 layers and TP strategy.")
        
        print(f"Number of layers: {len(model.layers)}")
        print(f"Rank: {rank}, World size: {world_size}")
        print(f"Device mesh: {strategy.device_mesh}")
        
        print("\nRunning forward pass...")
        x = torch.randint(0, config.src_vocab_size, (2, 16))
        with torch.no_grad():
            out = model(x)
        print(f"[Rank {rank}] Output shape: {out.shape}")
        
        if rank == 0 and wandb.run is not None:
            output_np = out.cpu().numpy()
            if not np.isnan(output_np).any():
                wandb.log({
                    "output_shape": str(out.shape),
                    "output_hist": wandb.Histogram(output_np),
                    "success": True
                })
            else:
                print("[Rank 0] Output contains NaNs — skipping histogram logging.")
                wandb.log({"success": False, "error": "Output contains NaNs"})
    
    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        if rank == 0 and wandb.run is not None:
            wandb.log({
                "success": False,
                "error": str(e),
                "error_trace": traceback.format_exc()
            })
        
        try:
            if wandb.run is not None:
                wandb.finish()
        except:
            pass
        
        try:
            dist.destroy_process_group()
        except:
            pass
        
        exit(1)
        
    if rank == 0 and wandb.run is not None:
        wandb.finish()
    
    try:
        dist.destroy_process_group()
    except:
        pass

if __name__ == "__main__":
    debug_tensor_parallel_strategy()