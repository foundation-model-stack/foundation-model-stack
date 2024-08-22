import collections
import os
import re
from collections import ChainMap
from collections.abc import Iterable
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch

from fms.modules.linear import get_linear_type
from fms.modules.tp import TPModule


__adapters: MutableMapping[str, MutableMapping[str, Callable[[Mapping], Mapping]]] = {}


def register_adapter(
    architecture: str,
    source: str,
    adapter: Callable[[Mapping], Mapping],
):
    """
    Registers a state dict adapter to be available to the (de) serialization
    API.

    Args:
    architecture: The name of the model architecture, e.g. 'llama'
    source: A label representing the format of the weights to be converted.
            E.g. 'hf'
    adapter: the class of the adapter. The class must accept one constructor
                parameter, which will be a state dict (`OrderedDict`)
    """
    sources: MutableMapping[str, Callable[[Mapping], Mapping]] = {}
    if architecture in __adapters:
        sources = __adapters[architecture]

    if source in sources:
        raise KeyError(
            f"Variant {source} already registered for architecture {architecture}"
        )

    sources[source] = adapter
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


def _legacy_attn_unfused_to_fused_adapter(orig_sd):
    """
    Legacy adapter for converting pre 0.0.6 unfused attn weights to fused attn weights
    """
    new_sd = {}
    removed_params = set()
    orig_keys = set(orig_sd.keys())
    for name in orig_keys:
        # if the name is part of removed_params, we no longer want to process it
        if name in removed_params:
            continue

        if "attn.query" in name or "attn.key" in name or "attn.value" in name:
            # weight_type denotes weight or bias
            weight_type = name.split(".")[-1]

            unfused_weights = [
                re.sub(
                    rf"attn.(query|key|value).{weight_type}",
                    f"attn.query.{weight_type}",
                    name,
                ),
                re.sub(
                    rf"attn.(query|key|value).{weight_type}",
                    f"attn.key.{weight_type}",
                    name,
                ),
                re.sub(
                    rf"attn.(query|key|value).{weight_type}",
                    f"attn.value.{weight_type}",
                    name,
                ),
            ]
            removed_params.update(unfused_weights)
            new_name = re.sub(
                rf"attn.(query|key|value).{weight_type}",
                f"attn.in_proj.qkv_fused.{weight_type}",
                name,
            )
            new_sd[new_name] = torch.cat(
                [orig_sd.pop(w) for w in unfused_weights], dim=0
            )
        else:
            new_sd[name] = orig_sd.pop(name)
    return new_sd


def _legacy_mlp_glu_unfused_to_fused_adapter(orig_sd):
    """
    Legacy adapter for converting pre 0.0.6 unfused mlp glu weights to fused mlp glu weights
    """
    new_sd = {}
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
            new_sd[new_name] = torch.cat(
                [orig_sd.pop(w) for w in unfused_weights], dim=0
            )
        else:
            new_sd[name] = orig_sd.pop(name)
    return new_sd


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
        return lambda x: x
    else:
        return __adapters[architecture][source]


def get_adapted(
    architecture: str, source: Optional[str], state_dict: Mapping[str, Any]
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
    adapted = adapter(state_dict)
    return adapted


# `models` imports each model class, causing models and adapters to be registered.
# down here to avoid circular dependencies.
from fms import models


def _get_safetensors_item(key, file: Path, device: torch.device) -> torch.Tensor:
    from safetensors import safe_open  # type: ignore[import-untyped]

    with torch.no_grad():
        with safe_open(
            file, framework="pt", device=str(device)
        ) as model_weights:  # type: ignore[attr-defined]
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
        raise ValueError(f"FSDP checkpoints can only be loaded into an FSDP model")
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
        elif source == "hf" or "gptq_hf" in source:
            glob_pattern_list = ["*.bin", "*.safetensors", "*.pt"]
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
    assert (
        len(checkpoints) > 0
    ), f"Can't find the requested checkpoint data at {model_path}"

    if checkpoint_sharding is not None and checkpoint_sharding != "layer":
        assert world_size == len(
            checkpoints
        ), f"Loading a {checkpoint_sharding}-sharded checkpoint with len={len(checkpoints)} but world size is {world_size}"

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
                torch.load(str(ckpt_path), mmap=True) for ckpt_path in checkpoints
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
    distributed_strategy: Optional[str] = None,
    checkpoint_sharding: Optional[str] = None,
    initial_device: torch.device = torch.device("cpu"),
    is_sd_unfused: bool = False,
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
    distributed_strategy: the kind of possibly-distributed model in which we
            intend to load these weights. E.g. tp, fsdp, None. Used for weight
            sharding.
    checkpoint_sharding: the sharding format of the checkpoint.
            E.g. layer, tp, fsdp. Used for weight sharding.
    initial_device: where the weights will be loaded from disk.
    """

    # 1. Get the adapter from checkpoint sd to fms sd
    # TODO: could create custom source based on the unfuse_strategy, and call corresponding adapter
    # TODO: adapter now always convert to fused... need a separate one or customize this
    # source="gptq_hf_unfused"  # !!! DEBUG: hardcode source
    adapter = _get_adapter(architecture, source)

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
            fms_partial_sd = adapter(partial_sd)
            unused_keys_partial = _load_partial_state_dict(model, fms_partial_sd, needs_tp_sharding)
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
    # TODO: we may return or print full set of unused_keys
    # should not raise error but a warning would be useful


def _copy_if_present(parameter, tensor_value):
    parameter.copy_(tensor_value, non_blocking=True)


def _move_to_real_device(param, real_device):
    if param.device == torch.device("meta"):
        if isinstance(param, torch.nn.Parameter):
            param = torch.nn.Parameter(
                torch.empty_like(param, device=real_device)
            )
        else:
            param = torch.empty_like(param, device=real_device)
    return param


def _load_partial_state_dict(
    model: torch.nn.Module, state_dict, needs_tp_sharding: bool
) -> set:
    unused_keys = set()
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

        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     breakpoint()

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
                    param = _move_to_real_device(param, tensor_value.device)
                    setattr(target_module, key_steps[-1], param)
                    param = getattr(target_module, key_steps[-1])

                param.copy_(tensor_value, non_blocking=True)
            elif tp_module is not None and tp_module not in seen_tp_modules:
                seen_tp_modules.add(tp_module)
                tensor_values = {k: v for k, v in state_dict.items() if tp_prefix in k}
                tp_module._apply(lambda t: _move_to_real_device(t, tensor_value.device))
                tp_module.load_weights(tensor_values)

        except AttributeError:
            # FIXME: error catch is incorrect, it will record `key` but the code
            # may have failed when processing one or more of this key's neighbors
            # (e.g., missing bias)
            unused_keys.add(key)

    return unused_keys
