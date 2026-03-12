"""
Helpers for working with the Hugging Face SmolVLM/Idefics3 processor.

We intentionally do not wrap HF processors in an FMS-specific class. Callers should use the
returned `transformers.AutoProcessor` directly.
"""

from __future__ import annotations

from typing import Any


def load_smolvlm_processor(
    checkpoint_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct", **kwargs: Any
):
    """
    Return the official Hugging Face processor for SmolVLM/Idefics3.

    This is a thin convenience helper around `transformers.AutoProcessor.from_pretrained`.
    """
    try:
        from transformers import AutoProcessor
    except ImportError as e:
        raise ImportError(
            "SmolVLM processor requires `transformers`. Install with: pip install transformers"
        ) from e

    return AutoProcessor.from_pretrained(checkpoint_name, **kwargs)
