"""
Ministral text generation example using FMS.

Usage (CPU, random weights — no download):
    python scripts/run_ministral.py --variant 3b
    python scripts/run_ministral.py --variant 8b

Usage (CPU, from HF-format checkpoint):
    python scripts/run_ministral.py --variant 3b --model_path /path/to/Ministral-3-3B-Instruct-2512-BF16
    python scripts/run_ministral.py --variant 8b --model_path /path/to/Ministral-3-8B-Instruct-2512-BF16

Usage (CUDA):
    python scripts/run_ministral.py --variant 3b --model_path /path/to/checkpoint --device cuda

To download checkpoints (requires `huggingface-cli` and access):
    huggingface-cli download mistralai/Ministral-3-3B-Instruct-2512-BF16 --local-dir /path/to/3b
    huggingface-cli download mistralai/Ministral-3-8B-Instruct-2512-BF16 --local-dir /path/to/8b

All HF-hosted Ministral variants are multimodal (Mistral3ForConditionalGeneration).
FMS extracts only the text-tower weights, discarding vision/projector layers.
The BF16 variants are recommended as they contain unquantized weights.

NOTE: transformers 4.57.6 does not recognise the 'ministral3' model_type, so we
load directly through FMS's own model registry with source="hf".
"""

import argparse
import os

import torch

from fms.models import get_model
from fms.utils.generation import generate


def main():
    parser = argparse.ArgumentParser(description="Ministral generation example")
    parser.add_argument(
        "--variant",
        type=str,
        default="3b",
        choices=["3b", "8b"],
        help="Model variant: 3b (3.4B params) or 8b (8.5B params)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to HF-format Ministral checkpoint directory. "
        "If omitted, a randomly initialised model is used (useful for smoke-testing).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"Loading Ministral {args.variant.upper()} on {args.device} ({args.dtype}) ...")
    if args.model_path is not None:
        # Use a glob pattern to load only the HF-sharded safetensors files.
        # The checkpoint directory also contains consolidated.safetensors
        # (Mistral native format) which would cause spurious key warnings.
        model_path = os.path.join(args.model_path, "*model*.safetensors")
        model = get_model(
            "mistral",
            args.variant,
            model_path=model_path,
            source="hf",
            device_type=args.device,
            data_type=dtype,
        )
    else:
        print("(no --model_path given, using random weights)")
        model = get_model(
            "mistral",
            args.variant,
            device_type=args.device,
            data_type=dtype,
        )

    model.eval()
    print(f"Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

    # --- tokenize ---
    # Ministral models use a SentencePiece / tekken tokenizer shipped with the
    # checkpoint.  When no real checkpoint is available we fall back to a simple
    # byte-level encoding so the script can still run end-to-end.
    ids = None
    if args.model_path is not None:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path, fix_mistral_regex=True
            )
            ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
        except Exception as e:
            print(f"Could not load tokenizer from {args.model_path}: {e}")

    if ids is None:
        # Fallback: encode each character as its ordinal (smoke-test only)
        ids = torch.tensor(
            [[ord(c) for c in args.prompt]], dtype=torch.long, device=args.device
        )
        tokenizer = None

    # --- generate ---
    print(f"\nPrompt: {args.prompt}")
    print("-" * 40)

    with torch.no_grad():
        result = generate(
            model,
            ids,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            do_sample=True,
            temperature=0.7,
        )

    # --- decode ---
    if tokenizer is not None:
        output_text = tokenizer.decode(result[0], skip_special_tokens=True)
    else:
        # Fallback: show the raw token IDs for the generated portion
        n_prompt = ids.shape[1]
        prompt_ids = result[0][:n_prompt].tolist()
        gen_ids = result[0][n_prompt:].tolist()
        prompt_text = "".join(chr(t) if t < 128 else "?" for t in prompt_ids)
        output_text = f"{prompt_text} [generated token ids: {gen_ids}]"

    print(output_text)


if __name__ == "__main__":
    main()
