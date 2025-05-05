import argparse
import os
import timeit
import statistics
import random
import torch
import numpy as np
from pathlib import Path

from fms import models
from fms.utils import tokenizers


def resolve_paths() -> dict:
    # Resolve base repo dir: .../foundation-model-stack
    script_path = Path(__file__).resolve()
    repo_dir = script_path.parents[2]
    model_dir = repo_dir.parent / "llama-hf"
    tokenizer_path = model_dir / "tokenizer.model"
    return {
        "repo_dir": repo_dir,
        "model_path": model_dir,
        "tokenizer_path": tokenizer_path,
        "env": "insomnia" if "insomnia001" in str(repo_dir) else "local"
    }


def parse_args(defaults):
    parser = argparse.ArgumentParser(description="Benchmark one-token ring vs regular attention")
    parser.add_argument("--device_type", type=str, default="cuda")
    parser.add_argument("--architecture", type=str, default="llama")
    parser.add_argument("--variant", type=str, default="7b")
    parser.add_argument("--model_path", type=str, default=str(defaults["model_path"]))
    parser.add_argument("--tokenizer", type=str, default=str(defaults["tokenizer_path"]))
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
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


def run_one_token(model, ids, label, repeat, device):
    def fn():
        logits = model.forward(ids, use_cache=False, only_last_token=True)
        _ = torch.argmax(logits, dim=-1)
        if device.type == "cuda":
            torch.cuda.synchronize()

    print(f"\n[Benchmark] {label}")
    results = timeit.repeat(fn, number=ids.shape[1], repeat=repeat)
    median = statistics.median(results)
    per_token = median / ids.shape[1] * 1000
    print(f"{label}: {per_token:.2f} ms/token")


def main():
    paths = resolve_paths()
    args = parse_args(paths)
    set_determinism()

    device = torch.device(args.device_type)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_default_dtype(torch.float16)

    tokenizer = tokenizers.get_tokenizer(args.tokenizer)
    ids = torch.randint(tokenizer.vocab_size(), (args.batch_size, args.seq_len), device=device)

    order = [("Regular Attention", None), ("Ring Attention", "ring")]
    if args.run_ring_first:
        order.reverse()

    print(f"[INFO] Running in environment: {paths['env']}")
    print(f"[INFO] Using model: {args.model_path}")
    print(f"[INFO] Using tokenizer: {args.tokenizer}")
    print(f"[INFO] Batch size: {args.batch_size}, Seq length: {args.seq_len}")

    for label, strategy in order:
        model = setup_model(args, strategy)
        run_one_token(model, ids, label, args.repeat, device)


if __name__ == "__main__":
    main()
