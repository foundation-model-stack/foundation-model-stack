"""
HuggingFace adapter utilities for loading vision, text, and tokenizer components.

Provides both dummy implementations for CPU testing and real HF loaders
for production use with actual checkpoints.

Note: Most of the helpers in this module are optional utilities and are not
required for the core FMS-native `idefics3` forward path. They exist to support
parity investigations and convenience loading of HF/SmolVLM components without
introducing a hard `transformers` dependency at import time.
"""

import logging
import torch
import torch.nn as nn
from typing import Any, Tuple

logger = logging.getLogger(__name__)


def load_tokenizer(name_or_path: str, image_token: str = "<image>"):
    """
    Load HuggingFace tokenizer and ensure image placeholder token exists.

    Args:
        name_or_path: HF model name or local path
        image_token: Special token for image placeholders (default: "<image>")

    Returns:
        Tokenizer with image_token in vocabulary
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )

    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)

    # Set padding token if not present (GPT-2 doesn't have one by default)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        logger.info("Set pad_token to eos_token (id=%s)", tok.pad_token_id)

    # Ensure image placeholder token exists
    if image_token not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [image_token]})
        logger.info(
            "Added special token %r to tokenizer (id=%s)",
            image_token,
            tok.convert_tokens_to_ids(image_token),
        )

    return tok


def load_text_backbone(
    name_or_path: str, device: str = "cpu", torch_dtype=torch.float32, tokenizer=None
):
    """
    Load HuggingFace causal language model for text generation.

    Args:
        name_or_path: HF model name or local path
        device: Target device (cpu/cuda)
        torch_dtype: Model dtype (default: float32 for CPU)
        tokenizer: Optional tokenizer to resize embeddings for new tokens

    Returns:
        HF model in eval mode
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        torch_dtype=torch_dtype,
        device_map=None,  # Manual device placement
    )

    # Resize token embeddings if tokenizer is provided and has new tokens
    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    model.to(device)
    return model


class DummySigLIPVision(nn.Module):
    """
    Dummy SigLIP vision encoder for CPU testing without real checkpoint.

    Mimics the output shape of SigLIP: (B, num_patches, hidden_dim)
    For 512x512 images with 16x16 patches: (B, 1024, 768)
    """

    def __init__(
        self, image_size: int = 512, patch_size: int = 16, hidden_dim: int = 768
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (image_size // patch_size) ** 2

        # Fake patch embedding projection
        self.proj = nn.Linear(3 * patch_size * patch_size, hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W) float tensor, values in [0, 1]

        Returns:
            Patch embeddings: (B, num_patches, hidden_dim)
        """
        B, C, H, W = images.shape
        assert H == self.image_size and W == self.image_size, (
            f"Expected {self.image_size}x{self.image_size} images, got {H}x{W}"
        )

        # Extract non-overlapping patches: unfold creates (B, C, H', W', patch_size, patch_size)
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )

        # Reshape to (B, num_patches, C * patch_size * patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(
            B, self.num_patches, C * self.patch_size * self.patch_size
        )

        # Project to hidden dimension
        patch_embeds = self.proj(patches)  # (B, num_patches, hidden_dim)

        return patch_embeds


def load_siglip_vision(
    name_or_path: str = "google/siglip-base-patch16-512",
    device: str = "cpu",
    use_dummy: bool = False,
) -> nn.Module:
    """
    Load SigLIP vision encoder or dummy for testing.

    Args:
        name_or_path: HF model name or local path (ignored if use_dummy=True)
        device: Target device
        use_dummy: If True, return DummySigLIPVision for CPU testing

    Returns:
        Vision encoder module returning (B, num_patches, hidden_dim)
    """
    if use_dummy:
        logger.info("Using DummySigLIPVision (no real checkpoint)")
        return DummySigLIPVision().to(device).eval()

    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )

    # Load real SigLIP vision model
    model = AutoModel.from_pretrained(name_or_path)
    model.eval()
    model.to(device)

    # Wrap to return just vision features
    class SigLIPWrapper(nn.Module):
        def __init__(self, siglip_model):
            super().__init__()
            self.vision_model = siglip_model.vision_model

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            """
            Args:
                images: (B, 3, H, W) preprocessed images

            Returns:
                Patch embeddings: (B, num_patches, hidden_dim)
            """
            # SigLIP returns last_hidden_state: (B, num_patches + 1, hidden_dim)
            # We exclude the CLS token (first position) for Idefics3
            outputs = self.vision_model(pixel_values=images)
            patch_embeds = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
            return patch_embeds

    return SigLIPWrapper(model).eval()


def load_internal_vision_tower(
    smolvlm_checkpoint_path: str, device: str = "cpu", normalize: bool = True
):
    """
    Load the internal vision tower from SmolVLM checkpoint.

    This extracts and wraps the vision encoder bundled inside SmolVLM,
    rather than loading a separate SigLIP model.

    Args:
        smolvlm_checkpoint_path: Path or HF model ID for SmolVLM checkpoint
        device: Target device
        normalize: If True, attempt to extract normalization params from checkpoint

    Returns:
        VisionTowerAdapter wrapping the internal tower

    Example:
        >>> from fms.models.idefics3_components.hf_adapter import load_internal_vision_tower
        >>> vision = load_internal_vision_tower(
        ...     "HuggingFaceTB/SmolVLM-256M-Instruct",
        ...     device="cpu"
        ... )
    """
    try:
        from transformers import AutoModel
        from .vision_adapter import (
            VisionTowerAdapter,
            extract_vision_tower_from_checkpoint,
        )
    except ImportError as e:
        raise ImportError(
            f"Required dependencies not available: {e}. "
            "Install with: pip install transformers"
        )

    logger.info("Loading SmolVLM checkpoint from: %s", smolvlm_checkpoint_path)
    checkpoint = AutoModel.from_pretrained(
        smolvlm_checkpoint_path, trust_remote_code=True
    )

    # Extract vision tower
    logger.info("Extracting internal vision tower...")
    inner_tower = extract_vision_tower_from_checkpoint(checkpoint)

    # TODO: Extract normalization parameters if normalize=True
    # Common patterns:
    #   - checkpoint.config.vision_config.image_mean
    #   - checkpoint.vision_tower.config.image_mean
    normalize_fn = None
    if normalize:
        # Attempt to extract mean/std
        try:
            if hasattr(checkpoint, "config"):
                vision_cfg = getattr(checkpoint.config, "vision_config", None)
                if vision_cfg is not None:
                    mean = getattr(vision_cfg, "image_mean", None)
                    std = getattr(vision_cfg, "image_std", None)
                    if mean is not None and std is not None:
                        from .vision_adapter import create_normalize_fn

                        normalize_fn = create_normalize_fn(mean, std, device)
                        logger.info("Using normalization: mean=%s, std=%s", mean, std)
        except Exception as e:
            logger.warning("Could not extract normalization params, using none: %s", e)

    # Wrap in adapter
    adapter = VisionTowerAdapter(inner_tower, normalize=normalize_fn)
    adapter.to(device)
    adapter.eval()

    logger.info("Vision tower loaded successfully (device=%s)", device)
    return adapter


def load_real_siglip_with_processor(
    name_or_path: str = "google/siglip-base-patch16-512",
):
    """
    Load SigLIP with image processor for end-to-end preprocessing.

    Returns:
        Tuple of (processor, vision_encoder_module)
    """
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )

    processor = AutoImageProcessor.from_pretrained(name_or_path)
    model = AutoModel.from_pretrained(name_or_path)

    class SigLIPWithProcessor(nn.Module):
        def __init__(self, proc, siglip_model):
            super().__init__()
            self.processor = proc
            self.vision_model = siglip_model.vision_model

        def preprocess(self, images):
            """
            Args:
                images: List of PIL Images or numpy arrays

            Returns:
                Preprocessed tensor ready for forward
            """
            inputs = self.processor(images=images, return_tensors="pt")
            return inputs["pixel_values"]

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            outputs = self.vision_model(pixel_values=images)
            patch_embeds = outputs.last_hidden_state[:, 1:, :]  # Remove CLS token
            return patch_embeds

    return processor, SigLIPWithProcessor(processor, model).eval()


def preprocess_image_simple(image, target_size: int = 512) -> torch.Tensor:
    """
    Simple CPU image preprocessing without requiring HF processor.

    Args:
        image: PIL Image or numpy array (H, W, 3)
        target_size: Output size (default 512 for SigLIP)

    Returns:
        Tensor (1, 3, target_size, target_size) normalized to [0, 1]
    """
    from PIL import Image
    import numpy as np

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Resize to target size
    image = image.convert("RGB").resize(
        (target_size, target_size), Image.Resampling.BILINEAR
    )

    # Convert to tensor and normalize to [0, 1]
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    return tensor


def _build_smolvlm_siglip_config_from_vision_config(vision_config):
    """
    Build SiglipVisionConfig to mirror SmolVLM HF vision_config.
    """
    from fms.models.siglip_vision import SiglipVisionConfig

    act = getattr(vision_config, "hidden_act", None)
    if act == "gelu_pytorch_tanh":
        act = "gelu-tanh"

    cfg = SiglipVisionConfig(
        image_size=getattr(vision_config, "image_size", 512),
        patch_size=getattr(vision_config, "patch_size", 16),
        num_channels=getattr(vision_config, "num_channels", 3),
        hidden_size=getattr(vision_config, "hidden_size", 768),
        intermediate_size=getattr(vision_config, "intermediate_size", 3072),
        nlayers=getattr(vision_config, "num_hidden_layers", 12),
        nheads=getattr(vision_config, "num_attention_heads", 12),
        hidden_act=act if act is not None else "gelu-tanh",
        layer_norm_eps=getattr(vision_config, "layer_norm_eps", 1e-6),
        attention_dropout=getattr(vision_config, "attention_dropout", 0.0),
    )
    return cfg


def _load_fms_siglip_from_smolvlm_vision(
    hf_vision_model, device: str = "cpu", dtype=torch.float32
):
    """
    Adapt HF SmolVLM vision weights to FMS SiglipVision and load (pooling head remains uninitialized).
    """
    from fms.models.siglip_vision import SiglipVision, _hf_to_fms_names, _weight_fusion

    vision_sd = hf_vision_model.state_dict()
    mapped_sd = _hf_to_fms_names(vision_sd)
    fused_sd = _weight_fusion(mapped_sd)
    fused_prefixed = {f"base_model.{k}": v for k, v in fused_sd.items()}

    cfg = _build_smolvlm_siglip_config_from_vision_config(hf_vision_model.config)
    siglip = SiglipVision(cfg).to(device=device, dtype=dtype)

    load_res = siglip.load_state_dict(fused_prefixed, strict=False)
    # SmolVLM HF vision tower has no classification/pooling head; head.* stays uninitialized
    # and will appear in missing_keys. We only use last_hidden_state via the wrapper.
    # Warn if anything beyond head.* is missing
    unexpected_missing = [k for k in load_res.missing_keys if not k.startswith("head.")]
    if unexpected_missing:
        logger.warning(
            "Unexpected missing keys in SiglipVision load: %s ...",
            unexpected_missing[:5],
        )

    if load_res.unexpected_keys:
        logger.warning(
            "Unexpected keys when loading SiglipVision: %s ...",
            load_res.unexpected_keys[:5],
        )

    return siglip


def load_smolvlm_checkpoint(
    checkpoint_path: str = "HuggingFaceTB/SmolVLM-256M-Instruct",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    use_fms_siglip_vision: bool = False,
) -> Tuple[Any, Any, Any, dict]:
    """
    Load full SmolVLM checkpoint and extract components for FMS model.

    Args:
        checkpoint_path: HF model ID or local path
        device: Target device

    Returns:
        Tuple of (tokenizer, vision_encoder, text_backbone, projector_weights)
        where projector_weights is a dict with 'weight' key for Idefics3Connector
    """
    try:
        from transformers import AutoModelForVision2Seq, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers library required for SmolVLM checkpoint loading"
        ) from e

    # Load full model (Idefics3ForConditionalGeneration)
    # We use AutoModelForVision2Seq to ensure we get the LM head
    full_model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    full_model.to(device=device, dtype=dtype)
    full_model.eval()
    full_sd = full_model.state_dict()

    # 1. Extract Vision Encoder
    # HF layout varies a bit across versions/wrappers. Prefer the canonical
    # `full_model.model.vision_model`, but fall back to a top-level `vision_model`
    # if present.
    vision_model = None
    if hasattr(full_model, "model") and hasattr(full_model.model, "vision_model"):
        vision_model = full_model.model.vision_model
    elif hasattr(full_model, "vision_model"):
        logger.info(
            "HF layout fallback: using full_model.vision_model (type=%s) because "
            "full_model.model.vision_model is missing",
            type(full_model).__name__,
        )
        vision_model = full_model.vision_model

    if vision_model is None:
        raise ValueError(
            f"Could not find vision_model in checkpoint (full_model type={type(full_model).__name__})"
        )

    # Wrap in adapter
    from .vision_adapter import (
        VisionTowerAdapter,
        create_normalize_fn,
        FmsSiglipVisionWrapper,
    )

    # Extract normalization if possible
    normalize_fn = None
    try:
        if hasattr(full_model.config, "vision_config"):
            mean = getattr(full_model.config.vision_config, "image_mean", None)
            std = getattr(full_model.config.vision_config, "image_std", None)
            if mean and std:
                normalize_fn = create_normalize_fn(mean, std, device)
    except Exception:
        pass

    if not use_fms_siglip_vision:
        vision_tower = vision_model
    else:
        vision_tower = FmsSiglipVisionWrapper(
            _load_fms_siglip_from_smolvlm_vision(
                vision_model, device=device, dtype=dtype
            )
        )

    vision_encoder = VisionTowerAdapter(vision_tower, normalize=normalize_fn)
    vision_encoder.to(device)

    # 2. Extract Connector Weights
    # Usually full_model.model.connector
    projector_weights = {}
    connector_module = None
    if hasattr(full_model, "model") and hasattr(full_model.model, "connector"):
        connector_module = full_model.model.connector

    if connector_module:
        for name, param in connector_module.named_parameters():
            # Strip 'modality_projection.' prefix if present (HF Idefics3 structure)
            new_name = name.replace("modality_projection.", "")
            projector_weights[new_name] = param.data.clone()
    else:
        # Fallback cases exist in the wild (e.g., different wrappers), but scanning named
        # parameters is risky and can silently pick the wrong tensors. Prefer to fail loudly
        # and let the caller use a known-good layout / loader path.
        logger.warning(
            "HF layout fallback: could not find full_model.model.connector (type=%s)",
            type(full_model).__name__,
        )
        raise ValueError(
            "Could not find connector module in HF checkpoint; cannot extract projector weights. "
            "Expected `full_model.model.connector` for Idefics3/SmolVLM."
        )

    # 3. Reconstruct Text Backbone (LlamaForCausalLM)
    # We need a functional CausalLM, but we only have the Idefics3 text submodule.
    # We will create a LlamaForCausalLM with the same config and load the weights.
    from transformers import LlamaForCausalLM, LlamaConfig

    text_config_dict = {}
    if hasattr(full_model.config, "text_config"):
        text_config_dict = full_model.config.text_config.to_dict()
    else:
        text_config_dict = full_model.config.to_dict()

    text_config_dict["vocab_size"] = full_model.config.vocab_size
    llama_config = LlamaConfig(**text_config_dict)
    text_backbone = LlamaForCausalLM(llama_config)

    text_sd = {}
    prefix_model = "model.text_model."
    prefix_head = "lm_head."
    for k, v in full_sd.items():
        if k.startswith(prefix_model):
            text_sd["model." + k[len(prefix_model) :]] = v
        elif k.startswith(prefix_head):
            text_sd[k] = v

    missing, unexpected = text_backbone.load_state_dict(text_sd, strict=False)
    if missing:
        logger.warning(
            "Missing keys in text backbone reconstruction: %s...", missing[:5]
        )

    text_backbone.to(device)
    text_backbone.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    return tokenizer, vision_encoder, text_backbone, projector_weights
