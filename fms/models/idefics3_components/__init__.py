"""
Internal implementation details for the `fms.models.idefics3` architecture.

This package is intentionally torch-only at import time (no `transformers`)
so that `import fms.models` remains lightweight.
"""

from .connector import Idefics3Connector
from .pack import pack_image_embeddings
from .fms_model import Idefics3, Idefics3Config
from .vision_adapter import (
    VisionTowerAdapter,
    FmsSiglipVisionWrapper,
    create_normalize_fn,
    extract_vision_tower_from_checkpoint,
)

__all__ = [
    "Idefics3Connector",
    "pack_image_embeddings",
    "Idefics3Config",
    "Idefics3",
    "VisionTowerAdapter",
    "FmsSiglipVisionWrapper",
    "create_normalize_fn",
    "extract_vision_tower_from_checkpoint",
]
