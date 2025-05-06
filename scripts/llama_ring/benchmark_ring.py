import argparse
import os
import statistics
import random
import warnings
import torch
import numpy as np
from pathlib import Path

import time # Use time module for manual timing
from fms import models
from fms.utils import tokenizers
from fms.distributed.strategy import NoOpStrategy # Import NoOpStrategy
import torch.distributed as dist # Import distributed module

from itertools import combinations

# Helper for printing only on rank 0
def print0(*args, **kwargs):
    # Get rank from env var if available, default to 0
    rank = int(os.getenv("RANK", 0))
    if rank == 0:
        print(*args, **kwargs)



def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark token generation with LLaMA attention strategies")

    # Resolve base repo dir: .../foundation-model-stack
    script_path = Path(__file__).resolve()
    repo_dir = script_path.parents[2]
    model_dir = repo_dir.parent / "llama-hf"
    tokenizer_path = model_dir / "tokenizer.model"

    parser.add_argument("--device_type", type=str, default="cuda")
    parser.add_argument("--architecture", type=str, default="llama")
    parser.add_argument("--variant", type=str, default="7b")
    parser.add_argument("--model_path", type=str, default=str(model_dir))
    parser.add_argument("--tokenizer", type=str, default=str(tokenizer_path), help="Full path to the tokenizer.model file")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=800, help="Prompt text len to use.")
    parser.add_argument("--num_tokens_to_benchmark", type=int, default=30, help="Number of tokens to generate and benchmark.")
    parser.add_argument("--run_ring_first", action="store_true", help="Explicitly run Ring Attention first (default). Set --no-run_ring_first to run Regular first.")
    parser.add_argument("--no-run_ring_first", dest="run_ring_first", action="store_false")
    parser.set_defaults(run_ring_first=True) # Default to Ring first

    return parser.parse_args()


def set_determinism():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # Note: Deterministic algorithms might not be fully supported with all operations/backends
    # torch.use_deterministic_algorithms(True)


def setup_model(args, strategy=None):
    # Map strategy string to actual strategy object if needed by get_model
    dist_strategy_param = strategy # Pass the strategy object ('ring' string or NoOpStrategy class)
    if strategy == "ring":
        # Ensure distributed is initialized before strategy instantiation attempt
        if not dist.is_initialized():
             # This should ideally not be reached if the main loop logic is correct
             raise RuntimeError("Attempted to setup Ring Attention without initialized torch.distributed")
    elif strategy is NoOpStrategy:
        # Pass the class type itself to get_model
        dist_strategy_param = NoOpStrategy

    model = models.get_model(
        args.architecture,
        args.variant,
        model_path=args.model_path,
        device_type=args.device_type,
        source="hf", # Assuming HF source
        distributed_strategy=dist_strategy_param,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def run_generation_benchmark(model, tokenizer, initial_ids, num_tokens_to_gen, label, device):
    """Generates tokens sequentially, timing and printing each one."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    print0(f"\n[Benchmark] Generating and timing {num_tokens_to_gen} tokens individually for '{label}' (Rank {rank})...")
    current_ids = initial_ids.clone()
    past_key_value_states = None
    token_times = []
    generated_tokens_text = [] # Store the text of generated tokens

    # Generation loop
    for i in range(num_tokens_to_gen):
        input_ids_step = current_ids if past_key_value_states is None else current_ids[:, -1:]

        if device.type == "cuda":
            torch.cuda.synchronize() # Sync before timing
        start_time = time.perf_counter()

        logits = model.forward(
            input_ids_step,
            past_key_value_states=past_key_value_states,
            use_cache=False # Use KV cache for efficient generation
        )

        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

        if device.type == "cuda":
            torch.cuda.synchronize() # Sync after forward pass
        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000
        token_times.append(duration_ms)

        # Decode generated token
        predicted_token_str = tokenizer.convert_ids_to_tokens([next_token_id.item()])
        predicted_text = tokenizer.convert_tokens_to_string(predicted_token_str)
        generated_tokens_text.append(predicted_text)

        # Update IDs for the next iteration
        current_ids = torch.cat([current_ids, next_token_id], dim=1)

    # Print summary statistics only on rank 0
    if rank == 0 and token_times:
        avg_time = statistics.mean(token_times)
        median_time = statistics.median(token_times)

        # Print the generated sequence first
        print0(f"\nGenerated Sequence ({label}): {'-'.join(generated_tokens_text)}")
        # Then print the timings as bullet points
        print0("Token Timings:")
        for i, duration_ms in enumerate(token_times):
            # Represent special characters like \n as \\n for cleaner printing
            printable_token = generated_tokens_text[i].encode('unicode_escape').decode('utf-8')
            print0(f"  * Token {i+1} ({printable_token}): {duration_ms:.2f} ms")
        print0(f"\nSummary for {label}:")
        print0(f"  Average time per token: {avg_time:.2f} ms")
        print0(f"  Median time per token: {median_time:.2f} ms")
        print0(f"  Total generation time for {num_tokens_to_gen} tokens: {sum(token_times):.2f} ms")

    return logits

def main():
    args = parse_args()
    set_determinism()

    # --- Initialize Distributed Environment ONLY if launched via torchrun/slurm ---
    # torchrun sets these env vars
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))

    distributed_backend = None
    if world_size > 1:
        if args.device_type == "cuda":
            if not torch.cuda.is_available():
                 raise EnvironmentError("CUDA requested but not available")
            torch.cuda.set_device(local_rank)
            distributed_backend = "nccl"
        else:
            distributed_backend = "gloo"

        if not dist.is_initialized():
             print(f"Initializing distributed process group (Rank {rank}/{world_size}) with backend '{distributed_backend}'...")
             dist.init_process_group(backend=distributed_backend)
    else:
        # Ensure rank is 0 if not distributed
        rank = 0


    device = torch.device(args.device_type, local_rank if local_rank != -1 and args.device_type == "cuda" else 0)
    torch.set_default_dtype(torch.float16)

    # Load tokenizer only on rank 0 to avoid redundant downloads/loads if applicable
    tokenizer = None
    if rank == 0:
        tokenizer = tokenizers.get_tokenizer(args.tokenizer)
    if world_size > 1:
        # Ensure rank 0 has loaded before others proceed
        dist.barrier()
        if rank != 0: # Load on other ranks after rank 0
             tokenizer = tokenizers.get_tokenizer(args.tokenizer)

    # Use tokenizer's pad_id if available, otherwise default to 0
    pad_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') and tokenizer.pad_id is not None else 0

    # Prepare input IDs - only rank 0 needs to print info and create tensor
    ids = None
    if rank == 0:
        if args.prompt_len:
            prompt = ", ".join([str(i) for i in range(0,args.prompt_len)])
            print0(f"[INFO] Using prompt: '{prompt}'")
            tokens = tokenizer.tokenize(prompt)
            prompt_ids_list = tokenizer.convert_tokens_to_ids(tokens)
            prompt_len = len(prompt_ids_list)
            print0(f"[INFO] Using prompt length: {prompt_len}")
            # Directly use the tokenized prompt without padding/truncation
            ids_tensor = torch.tensor(prompt_ids_list, dtype=torch.long).unsqueeze(0).repeat(args.batch_size, 1)
        else:
            # Fallback if no prompt is given (though the default prompt exists)
            # We need a length, let's default to a small value like 10 if --seq_len is gone
            prompt_len = 10
            print0(f"[INFO] No prompt specified, generating random token IDs for sequence length: {prompt_len}")
            ids_tensor = torch.randint(tokenizer.vocab_size(), (args.batch_size, prompt_len), dtype=torch.long)

        # Move tensor to device after creation
        ids = ids_tensor.to(device)

    if world_size > 1:
        # Broadcast the input tensor from rank 0 to all other ranks
        # Create a placeholder on other ranks first
        if rank != 0:
            # Need shape info from rank 0
            shape_list = [None] # Placeholder for shape tuple
            dist.broadcast_object_list(shape_list, src=0)
            ids_shape = shape_list[0]
            ids = torch.empty(ids_shape, dtype=torch.long, device=device)
        elif rank == 0: # Rank 0 needs to provide the shape
            dist.broadcast_object_list([ids.shape], src=0)

        # Use broadcast instead of broadcast_object_list for tensors
        dist.broadcast(ids, src=0)
        dist.barrier() # Ensure all ranks have the tensor

    # Define benchmark order (Default: Ring first)
    order = [("Ring Attention", "ring"), ("Regular Attention", NoOpStrategy)]
    if not args.run_ring_first:
        order.reverse()

    print0(f"[INFO] Using model: {args.model_path}")
    print0(f"[INFO] Using tokenizer: {args.tokenizer}")
    print0(f"[INFO] Batch size: {args.batch_size}, Input Seq length: {ids.shape[1]}") # Print actual length

    results = []
    for label, strategy in order:
        is_regular_run = (strategy is NoOpStrategy)
        should_run_this_rank = not (is_regular_run and rank != 0)

        # Skip Ring Attention if not running distributed
        if strategy == "ring" and not dist.is_initialized():
             print0(f"\n[WARNING] Skipping '{label}' because torch.distributed is not initialized.")
             print0("[INFO] This usually requires running with torchrun or within a Slurm job launched with torchrun.")
             continue

        model = None
        if should_run_this_rank:
            if is_regular_run:
                 print0(f"[INFO] Rank {rank} setting up Regular Attention benchmark...")
            else:
                 print(f"[INFO] Rank {rank} setting up Ring Attention benchmark...") # Print on all ranks for Ring

            # Suppress the specific warning about inv_freq keys
            warnings.filterwarnings("ignore", message=r"Keys from checkpoint \(adapted to FMS\) not copied into model:.*rotary_emb\.inv_freq")
            model = setup_model(args, strategy)
            warnings.resetwarnings() # Reset warnings filter after model setup

        # Barrier to ensure all ranks wait for model setup on active ranks
        if world_size > 1:
            dist.barrier()

        # Run benchmark only on ranks that loaded the model
        if should_run_this_rank:
            results.append(run_generation_benchmark(model, tokenizer, ids, args.num_tokens_to_benchmark, label, device))

            # Clean up model memory
            del model
            if args.device_type == "cuda":
                torch.cuda.empty_cache()

        # Barrier after each benchmark iteration to ensure synchronization before the next loop or exit
        if world_size > 1:
            dist.barrier()

    comparisons = list(combinations(results, 2))
    for (out1, out2) in comparisons:
        print(f"out1: shape:{out1.size()}, tensor:{out1}")
        print(f"out2: shape:{out2.size()}, tensor:{out2}")
        torch.testing.assert_close(out1, out2)

if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized(): # Clean up the process group
            dist.destroy_process_group()
