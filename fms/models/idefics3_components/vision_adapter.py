"""
Vision tower adapter for SmolVLM internal vision encoder.

Wraps the vision tower bundled inside SmolVLM checkpoint and provides
a consistent interface that returns (B, 1024, 768) patch embeddings.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class VisionTowerAdapter(nn.Module):
    """
    Wraps the SmolVLM-bundled vision tower to provide consistent interface.

    The adapter:
    - Accepts images as float tensors in [0,1] with shape (B, 3, 512, 512)
    - Applies optional normalization if provided
    - Validates output shape is exactly (B, 1024, 768)
    - Returns patch embeddings ready for Idefics3Connector

    Args:
        inner_tower: The actual vision encoder module from SmolVLM checkpoint
        normalize: Optional callable that applies normalization to input images
                  Example: lambda x: (x - mean) / std
        expected_num_patches: Expected number of patches (default: 1024 for 32x32)
        expected_hidden_dim: Expected hidden dimension (default: 768)
    """

    def __init__(
        self,
        inner_tower: nn.Module,
        normalize: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        expected_num_patches: int = 1024,
        expected_hidden_dim: int = 768,
    ):
        super().__init__()
        self.inner = inner_tower.eval()
        self.normalize = normalize
        self.expected_num_patches = expected_num_patches
        self.expected_hidden_dim = expected_hidden_dim

        # Freeze the vision tower for inference/fine-tuning text only
        for param in self.inner.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Process images through vision tower.

        Args:
            images: Input images (B, 3, 512, 512) as float in [0, 1]

        Returns:
            Patch embeddings (B, 1024, 768)

        Raises:
            ValueError: If output shape doesn't match expected dimensions
        """
        # Apply normalization if provided
        x = images
        if self.normalize is not None:
            x = self.normalize(x)

        # Forward through internal vision tower
        feats = self.inner(x)

        # Handle different vision tower output formats
        if hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state
        elif isinstance(feats, dict):
            if "last_hidden_state" in feats:
                feats = feats["last_hidden_state"]
            elif "pooler_output" in feats:
                raise ValueError(
                    "Vision tower returned pooler_output, need patch features"
                )
            else:
                raise ValueError(
                    f"Unknown vision tower output format keys: {feats.keys()}"
                )

        # Handle CLS token if present (e.g. 1025 tokens)
        if feats.shape[1] == self.expected_num_patches + 1:
            feats = feats[:, 1:, :]

        # Validate output shape
        if feats.dim() != 3:
            raise ValueError(
                f"Vision tower must return (B, T, C), got {feats.dim()}D tensor"
            )

        if feats.size(1) != self.expected_num_patches:
            raise ValueError(
                f"Expected {self.expected_num_patches} patches, got {feats.size(1)}"
            )

        if feats.size(2) != self.expected_hidden_dim:
            raise ValueError(
                f"Expected {self.expected_hidden_dim} dims, got {feats.size(2)}"
            )

        return feats


class FmsSiglipVisionWrapper(nn.Module):
    """
    Wrap FMS SiglipVision so VisionTowerAdapter can treat it like the HF vision_model.

    The forward returns a dict with 'last_hidden_state' matching HF expectations.
    """

    def __init__(self, siglip_vision: nn.Module):
        super().__init__()
        self.siglip_vision = siglip_vision

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        # SiglipVision returns (last_hidden_state, pooler_output)
        last_hidden_state, _ = self.siglip_vision(
            pixel_values,
            patch_attention_mask=patch_attention_mask,
            output_hidden_states=False,
            **kwargs,
        )
        return {"last_hidden_state": last_hidden_state}


def create_normalize_fn(mean: list, std: list, device: str = "cpu") -> Callable:
    """
    Create a normalization function for vision tower preprocessing.

    Args:
        mean: Per-channel mean values [R, G, B]
        std: Per-channel standard deviation values [R, G, B]
        device: Target device for tensors

    Returns:
        Normalization function that can be passed to VisionTowerAdapter

    Example:
        >>> # ImageNet normalization
        >>> normalize = create_normalize_fn(
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225]
        ... )
        >>> adapter = VisionTowerAdapter(tower, normalize=normalize)
    """
    mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def normalize(images: torch.Tensor) -> torch.Tensor:
        """Normalize images with precomputed mean and std."""
        return (images - mean_tensor.to(images.device)) / std_tensor.to(images.device)

    return normalize


def extract_vision_tower_from_checkpoint(checkpoint):
    """
    Extract vision tower from a SmolVLM/Idefics3-style checkpoint.

    HF layouts vary slightly across wrappers/versions. For Idefics3, the most common path is
    `checkpoint.model.vision_model`. This helper will fall back to a small set of other known
    attribute paths, and will log when a non-default path is used.

    Args:
        checkpoint: Loaded SmolVLM model object, or a dict-like wrapper that contains submodules

    Returns:
        Vision tower module

    Raises:
        ValueError: If vision tower cannot be found

    Known attribute paths:
        - checkpoint.model.vision_model (default)
        - checkpoint.model.vision_tower
        - checkpoint.vision_model
        - checkpoint.vision_tower
    """

    def _get_attr(obj, attr: str):
        if hasattr(obj, attr):
            value = getattr(obj, attr)
            if value is not None:
                return value
        return None

    # Default path: checkpoint.model.vision_model (no log; this is the expected layout).
    if hasattr(checkpoint, "model"):
        tower = _get_attr(checkpoint.model, "vision_model")
        if tower is not None:
            return tower

        # Fallback: checkpoint.model.vision_tower
        tower = _get_attr(checkpoint.model, "vision_tower")
        if tower is not None:
            logger.info(
                "Vision tower layout fallback: using checkpoint.model.vision_tower (type=%s)",
                type(checkpoint).__name__,
            )
            return tower

    # Fallback: top-level attributes
    for attr in ("vision_model", "vision_tower", "vision"):
        tower = _get_attr(checkpoint, attr)
        if tower is not None:
            logger.info(
                "Vision tower layout fallback: using checkpoint.%s (type=%s)",
                attr,
                type(checkpoint).__name__,
            )
            return tower

    # Try dictionary keys (dict-like wrapper containing submodules; not a raw state_dict)
    if isinstance(checkpoint, dict):
        for key in ("vision_model", "vision_tower", "vision"):
            if key in checkpoint:
                tower = checkpoint[key]
                if isinstance(tower, nn.Module):
                    logger.info(
                        "Vision tower layout fallback: using checkpoint[%r] (dict)", key
                    )
                    return tower

    # Not found
    raise ValueError(
        "Could not find vision tower in checkpoint. Tried: "
        "checkpoint.model.vision_model, checkpoint.model.vision_tower, checkpoint.vision_model, "
        "checkpoint.vision_tower, checkpoint.vision. "
        f"(checkpoint type={type(checkpoint).__name__})"
    )
