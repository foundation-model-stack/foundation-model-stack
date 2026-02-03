"""
Standard 512x512 Preprocessor for SmolVLM (Exact HF Parity)

Wraps the official Hugging Face processor to guarantee bit-for-bit identical
preprocessing with SmolVLM-256M/1.2B-Instruct.

Current SmolVLM models use a single 512x512 patch (no multi-patch / any-res).
This wrapper:
- Handles PIL / torch / numpy inputs safely
- Normalizes shapes to (B, N, C, H, W) -> future-proof for multi-patch releases
- Provides validation tools
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
import torch

_PIL_IMPORT_ERROR: ImportError | None = None
try:
    from PIL import Image  # type: ignore[import-not-found]
except ImportError as e:
    Image = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = e


def _require_pil():
    if Image is None:
        raise ImportError(
            "SmolVLMPreprocessor image conversion requires `Pillow`. "
            "Install with: pip install Pillow"
        ) from _PIL_IMPORT_ERROR
    return Image


def _to_pil(img_like):
    image_cls = _require_pil()

    if isinstance(img_like, torch.Tensor):
        x = img_like.detach().cpu()
        if x.dim() != 3:
            raise ValueError(
                f"Unsupported tensor shape: {x.shape}. Expected 3D (C,H,W) or (H,W,C)"
            )
        if x.shape[0] == 3:  # (C,H,W) -> (H,W,C)
            x = x.permute(1, 2, 0)
        elif x.shape[2] != 3:
            raise ValueError(
                f"Input tensor must have 3 channels (RGB), got shape {x.shape}"
            )
        if x.dtype.is_floating_point:
            x = (
                (x.clamp(0, 1) * 255).byte()
                if x.max() <= 1.0
                else x.clamp(0, 255).byte()
            )
        return image_cls.fromarray(x.numpy().astype(np.uint8)).convert("RGB")
    if isinstance(img_like, np.ndarray):
        a = img_like
        if a.ndim != 3:
            raise ValueError(
                f"Unsupported numpy array shape: {a.shape}. Expected 3D RGB."
            )
        if a.shape[2] != 3 and a.shape[0] == 3:
            a = np.transpose(a, (1, 2, 0))
        if a.dtype.kind == "f":
            a = (
                (np.clip(a, 0, 1) * 255).astype(np.uint8)
                if a.max() <= 1.0
                else np.clip(a, 0, 255).astype(np.uint8)
            )
        img = image_cls.fromarray(a.astype(np.uint8))
        return img.convert("RGB") if img.mode != "RGB" else img
    if isinstance(img_like, image_cls.Image):
        return img_like.convert("RGB") if img_like.mode != "RGB" else img_like
    raise TypeError(f"Unsupported image type: {type(img_like)}")


class SmolVLMPreprocessor:
    """
    Preprocessor for SmolVLM (Standard 512x512).

    Wraps HF AutoProcessor to ensure exact parity with the official implementation.
    Currently, SmolVLM uses a single 512x512 patch per image (no multi-patch tiling).

    Key features:
    - Resizes/crops images to 512x512
    - Normalizes to [-1, 1] range (SigLIP defaults)
    - Generates pixel_attention_mask
    - Compatible with HF AutoProcessor output format
    """

    def __init__(self, checkpoint_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"):
        """
        Initialize preprocessor using HF's AutoProcessor for exact parity.

        Args:
            checkpoint_name: HF checkpoint name for processor config
        """
        try:
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(
                "SmolVLMPreprocessor requires `transformers`. Install with: pip install transformers"
            ) from e

        self.processor = AutoProcessor.from_pretrained(checkpoint_name)
        self.image_processor = self.processor.image_processor

        # Cache config for reference
        self.num_patches = None  # Will be determined from actual output
        self.patch_size = 512
        self.normalization_mean = [0.5, 0.5, 0.5]
        self.normalization_std = [0.5, 0.5, 0.5]

    def preprocess(
        self,
        image: Union[
            Any,
            torch.Tensor,
            np.ndarray,
            List[Any],
            List[torch.Tensor],
            List[np.ndarray],
        ],
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Preprocess image(s) using HF's image processor.
        Accepts a single image or a list of images and normalizes output to (B, N, C, H, W).

        Returns a dict with pixel_values, pixel_attention_mask, and num_patches.
        """

        # Normalize to a list of PIL images
        if isinstance(image, list):
            pil_images = [_to_pil(x) for x in image]
        else:
            pil_images = [_to_pil(image)]

        processed = self.image_processor(images=pil_images, return_tensors="pt")

        pixel_values = processed["pixel_values"]

        # Handle pixel_attention_mask (may be missing depending on processor)
        if "pixel_attention_mask" in processed:
            pixel_attention_mask = processed["pixel_attention_mask"]
        else:
            H, W = pixel_values.shape[-2:]
            # Create a mask of ones matching batch (and patches if applicable) after normalization below
            pixel_attention_mask = torch.ones(
                pixel_values.shape[:-3] + (H, W),
                dtype=torch.long,
                device=pixel_values.device,
            )

        # Normalize shapes to (B, N, C, H, W)
        # HF can return (C, H, W), (B, C, H, W), or (B, N, C, H, W)
        if pixel_values.dim() == 5:
            # Already (B, N, C, H, W)
            pass
        elif pixel_values.dim() == 4:
            # (B, C, H, W) -> Treat as batch of single-patch images -> (B, 1, C, H, W)
            pixel_values = pixel_values.unsqueeze(1)

            # Mask should be (B, H, W) -> (B, 1, H, W)
            if pixel_attention_mask.dim() == 3:
                pixel_attention_mask = pixel_attention_mask.unsqueeze(1)
            elif pixel_attention_mask.dim() == 4:
                # If mask was already (B, 1, H, W) or similar, leave it?
                # Ideally it matches pixel_values structure.
                # If mask is (B, C, H, W) (unlikely for mask), we might have issues.
                # Assuming standard HF output (B, H, W) for 4D pixels.
                pass
        elif pixel_values.dim() == 3:
            # (C, H, W) -> Single image, single patch -> (1, 1, C, H, W)
            pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)

            # Mask (H, W) -> (1, 1, H, W)
            if pixel_attention_mask.dim() == 2:
                pixel_attention_mask = pixel_attention_mask.unsqueeze(0).unsqueeze(0)

        # Determine num_patches from actual output
        num_patches = pixel_values.shape[1]

        return {
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,
            "num_patches": num_patches,
        }

    def validate_parity(
        self,
        our_output: Dict[str, torch.Tensor],
        hf_output: Dict[str, torch.Tensor],
        tolerance: float = 1e-5,
    ) -> Dict[str, object]:
        """
        Validate that our preprocessing matches HF's output.

        Args:
            our_output: Output from our preprocess()
            hf_output: Output from HF processor
            tolerance: Max allowed difference

        Returns:
            Dict with validation results
        """
        results: Dict[str, object] = {}

        # Shape check
        our_pixels = our_output["pixel_values"]
        hf_pixels = hf_output["pixel_values"]

        results["shape_match"] = our_pixels.shape == hf_pixels.shape

        if results["shape_match"]:
            # Statistics check
            mean_diff = abs(our_pixels.mean().item() - hf_pixels.mean().item())
            std_diff = abs(our_pixels.std().item() - hf_pixels.std().item())
            max_abs_diff = (our_pixels - hf_pixels).abs().max().item()

            results["mean_close"] = mean_diff < tolerance
            results["std_close"] = std_diff < tolerance
            results["values_close"] = max_abs_diff < tolerance

            results["stats"] = {
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "max_abs_diff": max_abs_diff,
            }

        # Attention mask check
        if "pixel_attention_mask" in our_output and "pixel_attention_mask" in hf_output:
            our_mask = our_output["pixel_attention_mask"]
            hf_mask = hf_output["pixel_attention_mask"]
            results["mask_match"] = torch.equal(our_mask.long(), hf_mask.long())

        return results


def create_preprocessor(
    checkpoint_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
) -> SmolVLMPreprocessor:
    """Convenience function to create preprocessor."""
    return SmolVLMPreprocessor(checkpoint_name)
