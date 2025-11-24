import collections
import os
import re
import logging
from collections import ChainMap
from collections.abc import Iterable
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Set, Union

import torch

from fms.modules.tp import TPModule
from fms.utils.config import ModelConfig

logger = logging.getLogger(__name__)


__adapters: MutableMapping[
    str, MutableMapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]
] = {}
__adapter_steps: MutableMapping[
    str, MutableMapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]
] = {}


def register_adapter_step(
    architecture: str,
    adapter_name: str,
    adapter_step: Callable[[Mapping[str, Any]], Mapping[str, Any]],
):
    """
    Registers a state dict adapter step to be available to the (de) serialization
    API.

    Args:
    architecture: The name of the model architecture, e.g. 'llama'
    adapter_name: A name to identiy the step for the checkpoint and model
    adapter_step: the class of the adapter. The class must accept two constructor
        parameters, which will be a state dict (`OrderedDict`), and a collection
        of supported kwargs (such as model_config, etc.)
    """
    adapter_steps: MutableMapping[
        str, Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = {}
    if architecture in __adapter_steps:
        adapter_steps = __adapter_steps[architecture]

    if adapter_name in adapter_steps:
        raise KeyError(
            f"Adapter step {adapter_name} already registered for architecture {architecture}"
        )

    adapter_steps[adapter_name] = adapter_step
    __adapter_steps[architecture] = adapter_steps


def register_adapter(
    architecture: str,
    source: str,
    adapter_steps: list[str],
):
    """
    Registers a state dict adapter to be available to the (de) serialization
    API.

    Args:
    architecture: The name of the model architecture, e.g. 'llama'
    source: A label representing the format of the weights to be converted.
            E.g. 'hf'
    adapter_steps: a list of registered steps to build the basic adapter for an
            architecture and source combination. This can be augmented with extra
            steps if needed during call to _get_adapter()
    """
    sources: MutableMapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {}
    if architecture in __adapters:
        sources = __adapters[architecture]

    if source in sources:
        raise KeyError(
            f"Source {source} already registered for architecture {architecture}"
        )

    # Create a new base adapter for this source
    step_functions = [__adapter_steps[architecture][step] for step in adapter_steps]

    def adapter_fn(initial_sd: Mapping[str, Any], **extra_kwargs) -> Mapping[str, Any]:
        def reduce_fn(
            state_dict: Mapping[str, Any],
            step_func: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        ) -> Mapping[str, Any]:
            return step_func(state_dict, **extra_kwargs)

        return reduce(
            reduce_fn,
            step_functions,
            initial_sd,
        )

    sources[source] = adapter_fn
    __adapters[architecture] = sources


def extend_adapter(
    architecture: str,
    source: str,
    adapter_steps: list[str],
):
    """
    Extends an existing state dict adapter to the (de) serialization
    API.

    Args:
    architecture: The name of the model architecture, e.g. 'llama'
    source: A label representing the format of the weights to be converted.
            E.g. 'hf'
    adapter_steps: a list of registered steps to extend the adapter for an
            architecture and source combination. This can be augmented with extra
            steps if needed during call to _get_adapter()
    """
    sources: MutableMapping[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {}
    if architecture not in __adapters or source not in __adapters[architecture]:
        raise KeyError(
            f"Source {source} must already be registered for architecture {architecture}"
        )

    orig_adapter_fn = __adapters[architecture][source]

    # Create a new extended adapter for this source
    step_functions = [orig_adapter_fn] + [
        __adapter_steps[architecture][step] for step in adapter_steps
    ]

    def adapter_fn(initial_sd: Mapping[str, Any], **extra_kwargs) -> Mapping[str, Any]:
        def reduce_fn(
            state_dict: Mapping[str, Any],
            step_func: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        ) -> Mapping[str, Any]:
            return step_func(state_dict, **extra_kwargs)

        return reduce(
            reduce_fn,
            step_functions,
            initial_sd,
        )

    sources[source] = adapter_fn
    __adapters[architecture] = sources


def list_sources(architecture: str):
    """
    Lists available sources (attribute formats) of a model architecture.
    E.g. `models.list_variants('llama')` -> ['meta', 'fms', 'hf']
    Args:
    architecture: one of the registered architectures returned by
                    `models.list_models()`.
    """
    if architecture not in __adapters:
        return []
    return list(__adapters[architecture].keys())


def _pre006_attn_adapter_step(
    orig_sd: Mapping[str, Any], **kwargs
) -> Mapping[str, Any]:
    """
    Legacy adapter step for converting pre 0.0.6 unfused attn weights to fused attn weights
    """
    return _attn_unfused_to_fused(orig_sd, attn_prefix="attn")


def _attn_unfused_to_fused_step(
    orig_sd: Mapping[str, Any], **kwargs
) -> Mapping[str, Any]:
    """
    Adapter step for converting unfused attn weights to fused attn weights
    """
    return _attn_unfused_to_fused(orig_sd, attn_prefix="attn.in_proj")


def _attn_unfused_to_fused(
    orig_sd: Mapping[str, Any], attn_prefix: str
) -> Mapping[str, Any]:
    mutable_sd = dict(orig_sd)
    removed_params = set()
    orig_keys = set(orig_sd.keys())
    for name in orig_keys:
        # if the name is part of removed_params, we no longer want to process it
        if name in removed_params:
            continue

        if (
            f"{attn_prefix}.query" in name
            or f"{attn_prefix}.key" in name
            or f"{attn_prefix}.value" in name
        ):
            # weight_type denotes weight or bias
            weight_type = name.split(".")[-1]

            unfused_weights = [
                re.sub(
                    rf"{attn_prefix}.(query|key|value).{weight_type}",
                    f"{attn_prefix}.query.{weight_type}",
                    name,
                ),
                re.sub(
                    rf"{attn_prefix}.(query|key|value).{weight_type}",
                    f"{attn_prefix}.key.{weight_type}",
                    name,
                ),
                re.sub(
                    rf"{attn_prefix}.(query|key|value).{weight_type}",
                    f"{attn_prefix}.value.{weight_type}",
                    name,
                ),
            ]
            removed_params.update(unfused_weights)
            new_name = re.sub(
                rf"{attn_prefix}.(query|key|value).{weight_type}",
                f"attn.in_proj.qkv_fused.{weight_type}",
                name,
            )
            mutable_sd[new_name] = torch.cat(
                [mutable_sd.pop(w) for w in unfused_weights], dim=0
            )
    return mutable_sd


def _mlp_glu_unfused_to_fused_adapter_step(
    orig_sd: Mapping[str, Any], **kwargs
) -> Mapping[str, Any]:
    """
    Adapter step for converting unfused mlp glu weights to fused mlp glu weights
    """
    mutable_sd = dict(orig_sd)
    removed_params = set()
    orig_keys = set(orig_sd.keys())
    for name in orig_keys:
        # if the name is part of removed_params, we no longer want to process it
        if name in removed_params:
            continue

        if "ff_sub_layer.wg1_fused" not in name and (
            "ff_sub_layer.wg" in name or "ff_sub_layer.w1" in name
        ):
            weight_type = name.split(".")[-1]

            unfused_weights = [
                re.sub(
                    rf"ff_sub_layer.(wg|w1).{weight_type}",
                    f"ff_sub_layer.wg.{weight_type}",
                    name,
                ),
                re.sub(
                    rf"ff_sub_layer.(wg|w1).{weight_type}",
                    f"ff_sub_layer.w1.{weight_type}",
                    name,
                ),
            ]
            removed_params.update(unfused_weights)
            new_name = re.sub(
                rf"ff_sub_layer.(w1|wg).{weight_type}",
                f"ff_sub_layer.wg1_fused.{weight_type}",
                name,
            )
            mutable_sd[new_name] = torch.cat(
                [mutable_sd.pop(w) for w in unfused_weights], dim=0
            )
    return mutable_sd


def _get_adapter(
    architecture: str, source: Optional[str]
) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    if (
        source is None
        or architecture not in __adapters
        or source not in __adapters[architecture]
    ):
        # if no adapter is registered, assume the attributes are already in
        # fms format.
        # should we raise an error here instead?
        def id_fn(state_dict: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
            return state_dict

        return id_fn
    else:
        return __adapters[architecture][source]


def get_adapted(
    architecture: str,
    source: Optional[str],
    state_dict: Mapping[str, Any],
    adapter_kwargs: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Convert a state dict to FMS format, using an adapter specified by name.

    Args:
    architecture: one of the architectures from `models.list_models()`.
                    E.g. llama.
    source: A reference to an attribute format
    state_dict: the model.state_dict() to be converted/adapted.
    """
    # sometimes we only load onto rank 0 so may not have a state_dict here.
    if not len(state_dict):
        return state_dict
    adapter = _get_adapter(architecture, source)
    adapted = adapter(state_dict, **adapter_kwargs)
    return adapted


# `models` imports each model class, causing models and adapters to be registered.
# down here to avoid circular dependencies.


def _get_safetensors_item(key, file: Path, device: torch.device) -> torch.Tensor:
    from safetensors import safe_open  # type: ignore[import-untyped]

    with torch.no_grad():
        with safe_open(file, framework="pt", device=str(device)) as model_weights:  # type: ignore[attr-defined]
            return model_weights.get_tensor(key)


class LazySafetensorsDict(collections.UserDict):
    def set_lazy_tensor(self, key, file, device):
        super().__setitem__(key, lambda: _get_safetensors_item(key, file, device))

    def __getitem__(self, key):
        lazy_tensor = super().__getitem__(key)
        if callable(lazy_tensor):
            lazy_tensor = lazy_tensor()
            super().__setitem__(key, lazy_tensor)
        return lazy_tensor


def load_state_dict(
    model_path: Union[str, Path],
    *,
    source: Optional[str] = None,
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    initial_device: torch.device = torch.device("cpu"),
    rank: int = 0,
    world_size: int = 1,
) -> MutableMapping[str, Any]:
    """
    Validates that the file(s) found at a checkpoint path are compatible with
    the intended (possibly distributed) use-case, and returns a lazy loading
    state dict if possible (some formats may not support that).

    If model_path is a directory, it'll try to load models based on the source
    (e.g. .bin for HF, .pth for Meta), and, if no source is specified or hasn't
    been registered, it'll try .safetensors, .pth, and .bin.

    Args:
    model_path: the path to find the weights. If not set, return None.
    source: If the weights in the state dict didn't come from an FMS model,
            `source` specifies which conversion function might be needed.
            See `serialization.list_sources(architecture)`
    distributed_strategy: the kind of possibly-distributed model in which we
            intend to load these weights. E.g. tp, fsdp, None. Used for
            validation.
    checkpoint_sharding: the sharding format of the checkpoint.
            E.g. layer, tp, fsdp.
    initial_device: where the state dict will be loaded if not lazy.
            If meta, return empty dict.
    """
    if model_path is None or initial_device.type == "meta":
        return {}
    if checkpoint_sharding == "fsdp" and distributed_strategy not in ["fsdp", "hsdp"]:
        raise ValueError("FSDP checkpoints can only be loaded into an FSDP model")
    if checkpoint_sharding == "tp" and distributed_strategy != "tp":
        raise ValueError("TP checkpoints can only be loaded into a TP model")

    # Before creating the Path object, check if model_path has a glob pattern
    if isinstance(model_path, str):
        model_path, sep, glob_pattern = model_path.partition("*")
    else:
        sep = ""
        glob_pattern = ""
    glob_pattern = sep + glob_pattern

    model_path = Path(os.path.expanduser(model_path))

    checkpoints = []

    if model_path.is_dir():
        if glob_pattern != "":
            glob_pattern_list = [glob_pattern]
        elif source == "meta":
            glob_pattern_list = ["*.pth", "*.safetensors"]
        elif source == "hf":
            glob_pattern_list = ["*.safetensors", "*.bin", "*.pt"]
        else:
            glob_pattern_list = ["*.safetensors", "*.pth", "*.bin"]
        for glob_pattern_possibility in glob_pattern_list:
            file_list = list(model_path.glob(glob_pattern_possibility))
            if len(file_list) > 0:
                checkpoints = sorted(file_list)
                break

    if model_path.is_file():
        checkpoints = [model_path]

    # Check if we found some files
    assert len(checkpoints) > 0, (
        f"Can't find the requested checkpoint data at {model_path}"
    )

    if checkpoint_sharding is not None and checkpoint_sharding != "layer":
        assert world_size == len(checkpoints), (
            f"Loading a {checkpoint_sharding}-sharded checkpoint with len={len(checkpoints)} but world size is {world_size}"
        )

        checkpoints = [checkpoints[rank]]

    # if there's only one checkpoint for fsdp/hsdp, load it only into rank zero
    # and it will be distributed by the FSDP `sync_module_states` parameter
    if checkpoint_sharding is None and distributed_strategy in {"hsdp", "fsdp"}:
        if rank != 0:
            return {}

    checkpoint_sds = []
    if checkpoints[0].suffix == ".safetensors":
        for ckp in checkpoints:
            checkpoint_sds.append(
                _load_safetensors_state_dict(
                    ckp,
                    initial_device,
                )
            )
    else:
        with torch.no_grad():
            checkpoint_sds = [
                torch.load(str(ckpt_path), mmap=True, map_location=initial_device)
                for ckpt_path in checkpoints
            ]
    return ChainMap(*checkpoint_sds)


def _load_safetensors_state_dict(
    checkpoint: Path,
    device: torch.device,
):
    sd = LazySafetensorsDict()

    from safetensors import safe_open

    with safe_open(checkpoint, framework="pt", device=str(device)) as model_weights:  # type: ignore[attr-defined]
        sd_keys = list(model_weights.keys())
        for key in sd_keys:
            sd.set_lazy_tensor(key, checkpoint, device)
    return sd


def _find_key_neighbors(key: str, sd_keys: Set[str]):
    # For loading most models that concern us, a good partition is the
    # one used for FSDP units: everything that is in a layer can
    # go together, everything else can also go together and memory usage
    # will be keep in control.
    key_steps = key.split(".")
    prefix = ""
    # Navigate the model tree to find a layer index. If not found,
    # grab everything that is not numerical
    has_number = False
    for idx, step in enumerate(key_steps):
        prefix = ".".join(key_steps[: idx + 1])
        if step.isnumeric():
            prefix += "."
            has_number = True
            break
    prefix_neighbors = set()
    if has_number:
        for key_in_sd in sd_keys:
            if prefix in key_in_sd:
                prefix_neighbors.add(key_in_sd)
    else:
        for key_in_sd in sd_keys:
            if not bool(re.search(r"\.\d+\.", key_in_sd)):
                prefix_neighbors.add(key_in_sd)
    return list(prefix_neighbors)


def load_state_dict_into_model(
    model: torch.nn.Module,
    state_dict: MutableMapping[str, Any],
    architecture: str,
    source: str,
    dtype: Optional[torch.dtype] = None,
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    initial_device: torch.device = torch.device("cpu"),
    rank: int = 0,
) -> None:
    """
    This function loads state_dict into model in the most efficient way possible,
    and it removes all weights that have been used in model from state_dict
    in order to conserve memory.

    Args:
    model: The model where the weights are being loaded.
    state_dict: The dictionary with all the weights. If it has been mmaped
            (for torch.load) or it is an instance of LazySafetensorsDict,
            the weights are loaded lazily from disk.
    architecture: the model architecture, e.g. llama. See `models.list_models()`.
    source: If the weights in the state dict didn't come from an FMS model,
            `source` specifies which conversion function might be needed.
            See `serialization.list_sources(architecture)`
    dtype: If None, cast to state dict parameter data type when copying it
            into the model
    distributed_strategy: the kind of possibly-distributed model in which we
            intend to load these weights. E.g. tp, fsdp, None. Used for weight
            sharding.
    checkpoint_sharding: the sharding format of the checkpoint.
            E.g. layer, tp, fsdp. Used for weight sharding.
    initial_device: where the weights will be loaded from disk.
    """

    # 1. Get the adapter from checkpoint sd to fms sd
    adapter = _get_adapter(architecture, source)

    # Prepare the extra_kwargs for the adapter
    adapter_kwargs: dict[str, Any] = {}
    if hasattr(model, "config"):
        adapter_kwargs["model_config"] = model.config

    # 2. Decide if model needs sharding and how (for now only TP)
    needs_tp_sharding = checkpoint_sharding != "tp" and distributed_strategy == "tp"

    # 3. Iterate over the weights and load them into the model
    used_keys = set()
    unused_keys = set()
    sd_keys = set(state_dict.keys())

    with torch.no_grad():
        for key in sd_keys:
            if key in used_keys:
                continue
            used_keys.add(key)

            partial_sd = {key: state_dict[key]}
            # Find neighbors to the key. If the adapter requires a neighbor and
            # this function doesn't find it, it will crash.
            remaining_keys = sd_keys.difference(used_keys)
            neighbors = _find_key_neighbors(key, remaining_keys)
            for neighbor in neighbors:
                partial_sd[neighbor] = state_dict[neighbor]
                used_keys.add(neighbor)
            for psd_key in partial_sd.keys():
                if partial_sd[psd_key].device != initial_device:
                    partial_sd[psd_key] = partial_sd[psd_key].to(device=initial_device)
            fms_partial_sd = adapter(partial_sd, **adapter_kwargs)
            unused_keys_partial = _load_partial_state_dict(
                model=model,
                state_dict=fms_partial_sd,
                needs_tp_sharding=needs_tp_sharding,
                dtype=dtype,
            )
            unused_keys.update(unused_keys_partial)
            # Be aggressive in removing weights to save as much memory as possible
            for p_key in partial_sd.keys():
                if isinstance(state_dict, ChainMap):
                    for child_sd in state_dict.maps:
                        child_sd.pop(p_key, None)
                else:
                    state_dict.pop(p_key)
            del partial_sd
            del fms_partial_sd

    if unused_keys and rank == 0:
        logger.warning(
            f"Keys from checkpoint (adapted to FMS) "
            f"not copied into model: {unused_keys}"
        )


def _copy_if_present(parameter, tensor_value):
    parameter.copy_(tensor_value, non_blocking=True)


def _move_to_real_device(
    param: torch.Tensor,
    real_device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if param.device == torch.device("meta"):
        is_parameter = isinstance(param, torch.nn.Parameter)
        param = torch.empty_like(
            param,
            device=real_device,
            dtype=dtype,
        )
        if is_parameter:
            param = torch.nn.Parameter(param)
    return param


def _load_partial_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, Any],
    needs_tp_sharding: bool,
    dtype: Optional[torch.dtype] = None,
) -> set:
    unused_keys = set()
    unused_keys_tp = None
    seen_tp_modules = set()
    for key, tensor_value in state_dict.items():
        target_module = model
        # Find where to put the weight and decide whether it needs TP'ing
        key_steps = key.split(".")
        prefix = ""
        key_step = 0
        tp_module = None
        tp_prefix = ""

        # Navigate the model tree to find the module where the parameter is
        # located and whether there is a TPModule in the way in case the
        # parameter requires sharding
        while key_step < len(key_steps) - 1:
            try:
                target_module = getattr(target_module, key_steps[key_step])
                if key_step > 0:
                    prefix += "."
                prefix += key_steps[key_step]
                key_step += 1
                if isinstance(target_module, Iterable):
                    target_module = target_module[int(key_steps[key_step])]  # type: ignore[index]
                    prefix += "." + key_steps[key_step]
                    key_step += 1
                if isinstance(target_module, TPModule):
                    tp_module = target_module
                    tp_prefix = prefix
            except AttributeError:
                unused_keys.add(key)
                break

        # Check if target_module has the Parameter/buffer
        try:
            # If TP sharding is not needed, copy the parameter
            # into the model
            if not needs_tp_sharding or tp_module is None:
                param = getattr(target_module, key_steps[-1])

                # cast module parameter to non-meta device
                if param.device == torch.device("meta"):
                    param = _move_to_real_device(
                        param=param,
                        real_device=tensor_value.device,
                        dtype=tensor_value.dtype if dtype is None else dtype,
                    )
                    setattr(target_module, key_steps[-1], param)
                    param = getattr(target_module, key_steps[-1])
                param.copy_(tensor_value, non_blocking=True)

            elif tp_module is not None and tp_module not in seen_tp_modules:
                seen_tp_modules.add(tp_module)
                tensor_values = {k: v for k, v in state_dict.items() if tp_prefix in k}

                # when tensors from ckpt have all the same dtype,
                # it can be enforced onto the module parameters
                is_single_dtype = (
                    len(set([v.dtype for v in tensor_values.values()])) == 1
                )

                tp_module._apply(
                    lambda t: _move_to_real_device(
                        param=t,
                        real_device=tensor_value.device,
                        dtype=(
                            dtype
                            if dtype is not None
                            else tensor_value.dtype
                            if is_single_dtype
                            else t.dtype
                        ),
                    )
                )
                unused_keys_tp = tp_module.load_weights(tensor_values)
        except Exception as e:
            # capture error specific to shape mismatch and halt the processing
            if "shape" in str(e) or "size" in str(e):
                raise ValueError(
                    "Shape mismatch encountered while copying a tensor from the provided "
                    "checkpoint into the model.\nIf running a quantized model, it may "
                    "mean that the quantization setup used to train the checkpoint does "
                    "not match the one used to instantiate the model."
                ) from e
            if unused_keys_tp:
                unused_keys.update(unused_keys_tp)
            else:
                unused_keys.add(key)

    return unused_keys


# Expand QKV and Dense weights to match head_dim override
def _weight_expansion_for_mismatched_head_dim(
    input_sd: Mapping[str, Any], model_config
) -> Mapping[str, Any]:
    new_sd = dict(input_sd)

    # For multi model this expansion will be applicable only to the language_model
    if hasattr(model_config, "text_config") and isinstance(
        getattr(model_config, "text_config"), ModelConfig
    ):
        model_config = getattr(model_config, "text_config", model_config)
        if not all(["language_model" in layer for layer in input_sd]):
            return new_sd

    assert getattr(model_config, "head_dim", None) is not None, (
        "for weight expansion head_dim must be defined in model config"
    )

    # dimensions of layers to be expanded if needed
    layer_dim_div = {
        "attn.in_proj.query": (0, model_config.nheads),
        "attn.in_proj.key": (0, model_config.kvheads),
        "attn.in_proj.value": (0, model_config.kvheads),
        "attn.dense": (1, model_config.nheads),
    }

    # Computed head_dims for QKV and Dense
    head_dims_qkvd = [
        input_sd[layer].size(dim[0]) // dim[1]
        for layer in input_sd
        for tgt, dim in layer_dim_div.items()
        if tgt in layer
    ]

    # No attention layer in this step
    if len(set(head_dims_qkvd)) == 0:
        return new_sd

    # Computed head_dims for QKV and Dense should match
    assert len(set(head_dims_qkvd)) == 1, (
        "head_dims of QKV, and Dense layers do not agree"
    )

    assert model_config.head_dim % head_dims_qkvd[0] == 0, (
        f"weight expansion factor should not have fraction: {model_config.head_dim} / {head_dims_qkvd[0]}"
    )

    expansion_factor = model_config.head_dim // head_dims_qkvd[0]

    if expansion_factor > 1:
        assert expansion_factor % 2 == 0, "expansion factor must be an even number"

        expand_layer_dim = {
            layer: layer_dim_div[tgt][0]
            for layer in new_sd
            for tgt in layer_dim_div
            if tgt in layer
        }

        for layer, expand_dim in expand_layer_dim.items():
            tensor_value = new_sd[layer]
            original_size = list(tensor_value.size())
            expanded_size = original_size.copy()
            expanded_size[expand_dim] = expanded_size[expand_dim] * expansion_factor
            logger.warning(
                f"expanding weights of {('.'.join(layer.split('.')[1:-1])):30.30} {str(original_size):12.12} => {expanded_size}"
            )
            slices = [
                slice(0, None, expansion_factor) if dim == expand_dim else slice(None)
                for dim in range(tensor_value.ndim)
            ]
            # Assign the original weights tensor to the interleaved positions
            expanded_tensor = torch.zeros(expanded_size)
            expanded_tensor[tuple(slices)] = tensor_value
            new_sd[layer] = expanded_tensor

    return new_sd
