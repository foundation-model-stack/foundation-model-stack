from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from fms.modules.embedding import TPEmbedding
from fms.modules.positions import Alibi


# this probably belongs somewhere else but can't go in fms.distribtued b/c
# circular dependency.
def _tp_wrapped(module: nn.Module, group: ProcessGroup):
    if hasattr(module, "to_tp") and callable(module.to_tp):
        return module.to_tp(group)
    elif isinstance(module, Alibi):
        raise NotImplementedError("TODO: implement TP for Alibi")
        # tp_layer = TPAlibi.import_module(layer, world_size, rank, dtype)
        # setattr(model, name, tp_layer)
    elif isinstance(module, nn.Embedding):
        # We can't directly modify torch.nn modules to add the to_tp function
        return TPEmbedding.import_module(module, group)
    else:
        return module


def apply_tp(model: nn.Module, group: ProcessGroup):
    wrapped = _tp_wrapped(model, group)
    if wrapped is not model:
        return wrapped

    for name, layer in model.named_children():
        tp_layer = apply_tp(layer, group)
        setattr(model, name, tp_layer)
    return model
