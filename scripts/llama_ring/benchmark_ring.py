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
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="Periodic Table of Elements: \n * Hydrogen \n * Helium", help="Optional specific prompt text to use instead of random tokens.")
    parser.add_argument("--num_tokens_to_benchmark", type=int, default=10, help="Number of tokens to generate and benchmark.")
    parser.add_argument("--run_ring_first", action="store_true")
    return parser.parse_args()


def set_determinism():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def setup_model(args, strategy=None):
    model = models.get_model(
        args.architecture,
        args.variant,
        model_path=args.model_path,
        device_type=args.device_type,
        source="hf",
        distributed_strategy=strategy,
        fused_weights=True
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def run_generation_benchmark(model, tokenizer, initial_ids, num_tokens_to_gen, label, device):
    """Generates tokens sequentially, timing and printing each one."""
    print(f"\n[Benchmark] Generating and timing {num_tokens_to_gen} tokens individually for '{label}'...")
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

        logits, past_key_value_states = model.forward(
            input_ids_step,
            past_key_value_states=past_key_value_states,
            use_cache=True # Use KV cache for efficient generation
        )

        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

        if device.type == "cuda":
            torch.cuda.synchronize() # Sync after forward pass
        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000
        token_times.append(duration_ms)

        # Decode and print the generated token and its time
        predicted_token_str = tokenizer.convert_ids_to_tokens([next_token_id.item()])
        predicted_text = tokenizer.convert_tokens_to_string(predicted_token_str)
        generated_tokens_text.append(predicted_text)
        # Removed the print statement from inside the loop
        # Update IDs for the next iteration
        current_ids = torch.cat([current_ids, next_token_id], dim=1)

    # Print summary statistics
    if token_times:
        avg_time = statistics.mean(token_times)
        median_time = statistics.median(token_times)

        # Print the generated sequence first
        print(f"\nGenerated Sequence ({label}): {'-'.join(generated_tokens_text)}")
        # Then print the timings as bullet points
        print("Token Timings:")
        for i, duration_ms in enumerate(token_times):
            print(f"  * Token {i+1}: {duration_ms:.2f} ms")
        print(f"\nSummary for {label}:")
        print(f"  Average time per token: {avg_time:.2f} ms")
        print(f"  Median time per token: {median_time:.2f} ms")
        print(f"  Total generation time for {num_tokens_to_gen} tokens: {sum(token_times):.2f} ms")


def main():
    args = parse_args()
    set_determinism()

    device = torch.device(args.device_type)
    torch.set_default_dtype(torch.float16)

    tokenizer = tokenizers.get_tokenizer(args.tokenizer)
    # Use tokenizer's pad_id if available, otherwise default to 0
    pad_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') and tokenizer.pad_id is not None else 0

    if args.prompt:
        print(f"[INFO] Using prompt: '{args.prompt}'")
        # Tokenize the prompt using the FMS tokenizer's expected methods
        tokens = tokenizer.tokenize(args.prompt)
        prompt_ids_list = tokenizer.convert_tokens_to_ids(tokens)

        # Pad or truncate the prompt
        current_len = len(prompt_ids_list)
        target_len = args.seq_len + 10 # Define target_len here

        if current_len < target_len:
            padded_ids_list = prompt_ids_list + [pad_id] * (target_len - current_len)
            print(f"[INFO] Prompt padded to sequence length: {target_len}")
        elif current_len > target_len:
            padded_ids_list = prompt_ids_list[:target_len]
            print(f"[INFO] Prompt truncated to sequence length: {target_len}")
        else:
            padded_ids_list = prompt_ids_list
        ids = torch.tensor(padded_ids_list, dtype=torch.long, device=device).unsqueeze(0).repeat(args.batch_size, 1)
    else:
        print(f"[INFO] Generating random token IDs for sequence length: {args.seq_len}")
        ids = torch.randint(tokenizer.vocab_size(), (args.batch_size, args.seq_len), device=device)

    order = [("Regular Attention", None), ("Ring Attention", "ring")]
    if args.run_ring_first:
        order.reverse()

    print(f"[INFO] Using model: {args.model_path}")
    print(f"[INFO] Using tokenizer: {args.tokenizer}")
    print(f"[INFO] Batch size: {args.batch_size}, Seq length: {args.seq_len}")

    for label, strategy in order:
        # Skip Ring Attention if not in a distributed environment
        if strategy == "ring" and not torch.distributed.is_initialized():
            print(f"\n[INFO] Skipping '{label}' benchmark as torch.distributed is not initialized.")
            print("[INFO] Run with torchrun for distributed benchmarks like Ring Attention.")
            continue

        # Suppress the specific warning about inv_freq keys
        # Apply filter right before the operation that causes the warning
        warnings.filterwarnings("ignore", message=r"Keys from checkpoint \(adapted to FMS\) not copied into model:.*rotary_emb\.inv_freq")
        model = setup_model(args, strategy)
        # Reset warnings filter immediately after so other warnings are not suppressed
        warnings.resetwarnings() # Reset warnings filter after model setup

        run_generation_benchmark(model, tokenizer, ids, args.num_tokens_to_benchmark, label, device)


if __name__ == "__main__":
    main()
