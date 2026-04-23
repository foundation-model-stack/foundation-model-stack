"""
Public entrypoint for the `idefics3` architecture.

The implementation lives under `fms.models.idefics3_components` and is considered internal.
"""

import logging
import re
import warnings

from fms.models.idefics3_components import (
    Idefics3,
    Idefics3Config,
    Idefics3Connector,
    VisionTowerAdapter,
    create_normalize_fn,
    extract_vision_tower_from_checkpoint,
    pack_image_embeddings,
)
from fms.models.idefics3_components.preprocessing import load_smolvlm_processor
from fms.utils import serialization

logger = logging.getLogger(__name__)

__all__ = [
    "Idefics3Config",
    "Idefics3Connector",
    "Idefics3",
    "VisionTowerAdapter",
    "create_normalize_fn",
    "extract_vision_tower_from_checkpoint",
    "pack_image_embeddings",
    "load_smolvlm_preprocessor",
    "load_smolvlm_processor",
]

_architecture_name = "idefics3"


def load_smolvlm_preprocessor(*args, **kwargs):
    """
    Deprecated: use `load_smolvlm_processor` (returns HF AutoProcessor) instead.

    Backwards-compatibility note:
    Historically this returned an FMS wrapper with a `.preprocess()` method. To avoid
    breaking older call sites, we still return the HF processor, but attach a minimal
    `.preprocess()` helper that returns a normalized dict of tensors.
    """
    warnings.warn(
        "load_smolvlm_preprocessor is deprecated; use load_smolvlm_processor instead",
        FutureWarning,
        stacklevel=2,
    )
    processor = load_smolvlm_processor(*args, **kwargs)

    # Preserve legacy `.preprocess()` call sites by attaching a method to the processor.
    # This avoids re-introducing a public wrapper class while still being compatible.
    if not hasattr(processor, "preprocess"):
        import torch

        def _preprocess(image=None, images=None, **call_kwargs):
            if images is None:
                images = image
            if images is None:
                raise ValueError(
                    "Missing required image input. Provide `image=...` or `images=...`."
                )

            call_kwargs.setdefault("return_tensors", "pt")
            processed = processor(images=images, **call_kwargs)
            pixel_values = processed["pixel_values"]
            if not isinstance(pixel_values, torch.Tensor):
                raise TypeError(
                    f"Expected processor(..., return_tensors='pt') to return torch pixel_values, got {type(pixel_values)!r}"
                )

            if "pixel_attention_mask" in processed:
                pixel_attention_mask = processed["pixel_attention_mask"]
            else:
                H, W = pixel_values.shape[-2:]
                pixel_attention_mask = torch.ones(
                    pixel_values.shape[:-3] + (H, W),
                    dtype=torch.long,
                    device=pixel_values.device,
                )

            # Normalize shapes to (B, N, C, H, W)
            if pixel_values.dim() == 5:
                if (
                    isinstance(pixel_attention_mask, torch.Tensor)
                    and pixel_attention_mask.dim() == 3
                ):
                    # (B, H, W) -> (B, 1, H, W)
                    pixel_attention_mask = pixel_attention_mask.unsqueeze(1)
                if (
                    isinstance(pixel_attention_mask, torch.Tensor)
                    and pixel_attention_mask.dim() == 4
                    and pixel_attention_mask.shape[1] == 1
                    and pixel_values.shape[1] > 1
                ):
                    # (B, 1, H, W) -> (B, N, H, W)
                    pixel_attention_mask = pixel_attention_mask.expand(
                        pixel_values.shape[0],
                        pixel_values.shape[1],
                        pixel_attention_mask.shape[2],
                        pixel_attention_mask.shape[3],
                    ).contiguous()
            elif pixel_values.dim() == 4:
                # (B, C, H, W) -> (B, 1, C, H, W)
                pixel_values = pixel_values.unsqueeze(1)
                if (
                    isinstance(pixel_attention_mask, torch.Tensor)
                    and pixel_attention_mask.dim() == 3
                ):
                    # (B, H, W) -> (B, 1, H, W)
                    pixel_attention_mask = pixel_attention_mask.unsqueeze(1)
            elif pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)
                if (
                    isinstance(pixel_attention_mask, torch.Tensor)
                    and pixel_attention_mask.dim() == 2
                ):
                    pixel_attention_mask = pixel_attention_mask.unsqueeze(0).unsqueeze(
                        0
                    )

            num_patches = pixel_values.shape[1]
            return {
                "pixel_values": pixel_values,
                "pixel_attention_mask": pixel_attention_mask,
                "num_patches": num_patches,
            }

        processor.preprocess = _preprocess

    return processor


def _components_factory(**kwargs):
    return Idefics3(**kwargs)


def _smolvlm_factory(**kwargs):
    # This factory only instantiates the architecture; weights are loaded via the
    # standard FMS serialization pipeline (`source="hf"`).
    return Idefics3(**kwargs)


def _hf_to_fms_state_dict(input_sd, **kwargs):
    """
    Convert an HF Idefics3/SmolVLM checkpoint state_dict into the FMS idefics3 layout.

    Note: `model_config` is supplied via `serialization.load_state_dict_into_model()`
    adapter kwargs (i.e., `model.config`).
    """
    model_config = kwargs.get("model_config")
    if model_config is None:
        raise ValueError("Missing required adapter kwarg: model_config")
    if not isinstance(model_config, Idefics3Config):
        raise TypeError(
            f"Expected model_config to be Idefics3Config, got {type(model_config)!r}"
        )

    new_sd: dict = {}

    # ---- vision
    vision_sd = {}
    for k, v in input_sd.items():
        if k.startswith("model.vision_model."):
            vision_sd[k[len("model.") :]] = v  # keep vision_model.* prefix
        elif k.startswith("vision_model."):
            vision_sd[k] = v
    if vision_sd:
        adapted_vision = serialization.get_adapted(
            architecture="siglip_vision",
            source="hf",
            state_dict=vision_sd,
            adapter_kwargs={"model_config": model_config.vision_config},
        )
        for k, v in adapted_vision.items():
            new_sd[f"vision_tower.{k}"] = v

    # ---- text (re-key SmolVLM's text weights into an HF LlamaForCausalLM keyspace)
    text_hf_sd = {}
    for k, v in input_sd.items():
        if k.startswith("model.text_model."):
            text_hf_sd["model." + k[len("model.text_model.") :]] = v
        elif k.startswith("lm_head."):
            text_hf_sd[k] = v
    if text_hf_sd:
        adapted_text = serialization.get_adapted(
            architecture="llama",
            source="hf",
            state_dict=text_hf_sd,
            adapter_kwargs={"model_config": model_config.text_config},
        )
        for k, v in adapted_text.items():
            new_sd[f"text_model.{k}"] = v

    # ---- connector
    connector_replacements = [
        # Allow both "model.connector.*" and "connector.*"
        (r"^model\.", ""),
        # HF SmolVLM/Idefics3 uses: connector.modality_projection[.proj].* -> FMS connector.proj.*
        (r"^connector\.modality_projection\.proj\.", "connector.proj."),
        (r"^connector\.modality_projection\.", "connector.proj."),
        (r"^connector\.modality_projection$", "connector.proj.weight"),
    ]

    for name, param in input_sd.items():
        if not (name.startswith("connector.") or name.startswith("model.connector.")):
            continue

        new_name = name
        for pattern, repl in connector_replacements:
            new_name = re.sub(pattern, repl, new_name)

        if new_name.startswith("connector."):
            new_sd[new_name] = param

    return new_sd


def _register():
    from fms import models

    models.register_model(_architecture_name, "smolvlm-256m-instruct", _smolvlm_factory)
    models.register_model(_architecture_name, "components", _components_factory)


_register()


# ---- HF serialization adapters
serialization.register_adapter_step(
    _architecture_name, "hf_to_fms", _hf_to_fms_state_dict
)
serialization.register_adapter(_architecture_name, "hf", ["hf_to_fms"])
