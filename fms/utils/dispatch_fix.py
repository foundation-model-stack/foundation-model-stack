from torch import Tensor
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch._functorch._aot_autograd.schemas import ViewAndMutationMeta
import torch._functorch._aot_autograd.subclass_utils

def requires_subclass_dispatch_fixed(args, fw_metadata: ViewAndMutationMeta) -> bool:
    print("Running fixed")
    args_flattened = pytree.arg_tree_leaves(*args)
    any_subclass_args = any(
        is_traceable_wrapper_subclass(x)
        for x in args_flattened
        if isinstance(x, Tensor)
    )
    from torch._functorch._aot_autograd.schemas import SubclassCreationMeta

    any_subclass_outputs = any(
        type(x) is SubclassCreationMeta for x in fw_metadata.subclass_fw_graph_out_meta
    )
    # This tells us whether or not we need to perform any unwrapping/wrapping of tensor subclasses at runtime.
    return any_subclass_args or any_subclass_outputs


torch._functorch._aot_autograd.subclass_utils.requires_subclass_dispatch = requires_subclass_dispatch_fixed