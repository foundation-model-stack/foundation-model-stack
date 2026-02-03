import logging
from typing import Optional

from fms.models.idefics3_components import (
    Idefics3,
    Idefics3Config,
    Idefics3Connector,
    VisionTowerAdapter,
    create_normalize_fn,
    extract_vision_tower_from_checkpoint,
    pack_image_embeddings,
)
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
]

_architecture_name = "idefics3"


def load_smolvlm_preprocessor(*args, **kwargs):
    """
    Lazily import and construct the HF-backed SmolVLM preprocessor.

    This keeps `import fms.models` free of a hard dependency on `transformers`.
    """
    from fms.models.idefics3_components.preprocessing import SmolVLMPreprocessor

    return SmolVLMPreprocessor(*args, **kwargs)


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
    def _remap_connector_key(key: str) -> Optional[str]:
        if key.startswith("model."):
            key = key[len("model.") :]
        if not key.startswith("connector."):
            return None
        key = key[len("connector.") :]
        # HF SmolVLM uses: modality_projection.proj.weight -> FMS connector.proj.weight
        if key.startswith("modality_projection.proj."):
            key = key[len("modality_projection.proj.") :]
            return f"connector.proj.{key}"
        if key.startswith("modality_projection."):
            key = key[len("modality_projection.") :]
            return f"connector.proj.{key}"
        if key == "modality_projection":
            return "connector.proj.weight"
        return f"connector.{key}"

    for k, v in input_sd.items():
        remapped_key = _remap_connector_key(k)
        if remapped_key is not None:
            new_sd[remapped_key] = v

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
