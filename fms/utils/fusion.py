import torch
import torch.nn as nn


def _fusion_converted(module: nn.Module, fuse: bool):
    if not fuse and hasattr(module, "unfuse"):
        result = module.unfuse()
        del module
    elif fuse and hasattr(module, "fuse"):
        result = module.fuse()
        del module
    else:
        result = module
    return result


def apply_fusion(module: nn.Module, fuse: bool) -> nn.Module:
    """When applied to a module, will fuse/unfuse modules that support the fuse/unfuse method based

    Parameters
    ----------
    module: nn.Module
        the module to fuse/unfuse
    fuse: bool
        if True, will fuse any modules that support the fuse method, otherwise will unfuse any modules that support
        the unfuse method

    Returns
    -------
    nn.Module
        the original module fused/unfused
    """
    with torch.no_grad():
        wrapped = _fusion_converted(module, fuse)
        if wrapped is not module:
            return wrapped

        for name, layer in module.named_children():
            fused_layer = apply_fusion(layer, fuse)
            setattr(module, name, fused_layer)
        return module
