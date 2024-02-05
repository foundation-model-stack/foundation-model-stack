import contextlib
from typing import Dict
import functools

import torch
from torch._dynamo import allow_in_graph
from torch._guards import detect_fake_mode
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode

_HANDLED_FUNCTIONS = {}


def _implements(torch_function):
    """Register a torch function override"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class ExpandableTensor(torch.Tensor):
    """
    This tensor behaves similarly to a java ArrayList along a specified
    dimension. It preallocates space along that dimension, and has an append
    operation that utilizes this space. This can be more efficient than
    performing many consecutive torch.cat operations.

    When preallocated space is exhasted, the internal length is doubled.
    All operations performed on this tensor use a view that truncates the
    un-utilized preallocated space.

    This class overrides and deviates from the contract of `torch.cat` such
    that in some cases the result of a `torch.cat( (expandable, other) )` will
    be an in-place modified `expandable`. This could cause bugs in cases where
    the original tensor is modified in-place.

    Args:
        tensor: the initial values to hold in this tensor.
        dim: the expandable dimension
        preallocate_length: the total amount of space to allocate along
            dimension `dim`
    """

    _underlying_tensor: torch.Tensor
    _dim: int
    _dim_length: int

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, *, dim: int = 0, preallocate_length: int = None, dim_length=None):
        sizes = list(tensor.size())
        sizes[dim] = preallocate_length if preallocate_length is not None else sizes[dim]
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            sizes,
            tensor.stride(),
            tensor.storage_offset(),
            None,
            tensor.dtype,
            tensor.layout,
            tensor.device,
            False,
            False,
            "sizes",
            False,
            False,
        )
        return r

    def __init__(self, tensor, *, dim=0, preallocate_length=None, dim_length=None):
        self._dim = dim
        self._dim_length = tensor.shape[dim] if dim_length is None else dim_length
        self._underlying_tensor = tensor
        if preallocate_length is not None and preallocate_length > self._underlying_tensor.shape[dim]:
            sizes = list(tensor.size())
            sizes[dim] = preallocate_length - sizes[dim]
            reshape_sizes = [-1 for _ in sizes]
            reshape_sizes[dim] = sizes[dim]
            # fake_mode = detect_fake_mode(tensor)
            # cm = contextlib.nullcontext() if fake_mode is None else fake_mode
            # with cm:
            cat_slice = torch.index_select(tensor, dim, torch.tensor([self._dim_length-1], dtype=torch.long, device=tensor.device)).expand(*reshape_sizes)
            self._underlying_tensor = torch.cat((self._underlying_tensor, cat_slice), dim=dim)
        torch._dynamo.mark_dynamic(self, self._dim)

    # @_implements(torch.ops.aten.sym_size.default)
    # def sym_size(self, dim=None):
    #     return None

    
    @_implements(torch.ops.aten.size.default)
    def size(self, dim=None):
        # https://github.com/pytorch/pytorch/issues/111944
        sizes_list = list(self._underlying_tensor.size())
        sizes_list[self._dim] = self._dim_length
        if dim is None:
            return torch.Size(sizes_list)
        else:
            return sizes_list[dim]

    def _append(self, tensor):
        """
        Returns a tensor equivalent to the result of
        `torch.cat( (self, tensor), dim=self._dim)`, possibly modifying `self`
        in-place to make use of preallocated space.
        """
        dim = self._dim
        expected = list(self._underlying_tensor.size())
        tensor_sizes = list(tensor.size())
        for i in range(len(expected)):
            if i != dim:
                assert expected[i] == tensor_sizes[i]
        if self.size()[dim] + tensor.size()[dim] <= self._underlying_tensor.size()[dim]:
            # copy into tail of _tensor
            view = self._underlying_tensor
            sizes = list(view.size())
            sizes[self._dim] = tensor.size()[dim]
            strides = self._underlying_tensor.stride()
            offset = self._dim_length * strides[self._dim]
            view = view.as_strided(size=sizes, stride=strides, storage_offset=offset)
            view.copy_(tensor)
            result = ExpandableTensor(self._underlying_tensor, dim=self._dim)
            result._dim_length = self._dim_length + tensor.shape[dim]
            return result
        else:
            # create new expandable tensor
            expanded = ExpandableTensor(
                self._tensor(), dim=self._dim, preallocate_length=self._underlying_tensor.shape[dim] * 2
            )
            return expanded._append(tensor)

    def _tensor(self):
        """
        Returns a view of the tensor excluding preallocated space
        """
        view = self._underlying_tensor.view_as(self._underlying_tensor)
        sizes = list(view.size())
        sizes[self._dim] = self._dim_length
        view = view.as_strided(size=sizes, stride=view.stride())
        return view

    def __repr__(self):
        return f"ExpandableTensor(dim: {self._dim}, allocated_length: {self._underlying_tensor.size(self._dim)}, used_length: {self._dim_length}): " + self._tensor().__repr__()

    def __tensor_flatten__(self):
        ctx = {
            "_dim": self._dim,
            "_dim_length": self._dim_length,
        }
        inner_tensors = ["_underlying_tensor"]
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride):
        assert len(inner_tensors) == 1
        _underlying_tensor = inner_tensors["_underlying_tensor"]
        _dim = meta["_dim"]
        _dim_length = meta["_dim_length"]

        # Note that we cannot simply check if is_fake(values) because
        # during aot autograd, FunctionalTensors are not fake but hold
        # symbolic sizes.
        return ExpandableTensor(
            _underlying_tensor,
            dim=_dim,
            dim_length=outer_size[_dim],
        )

    @_implements(torch.ops.aten.cat.default)
    def cat(tensors, dim=0, *, out=None):
        if (
            len(tensors)
            and type(tensors[0]) == ExpandableTensor
            and tensors[0]._dim == dim
        ):
            result = tensors[0]
            for tensor in tensors[1:]:
                result = result._append(tensor)
            return result
        else:
            tensors = [
                tensor._tensor() if type(tensor) == ExpandableTensor else tensor
                for tensor in tensors
            ]
            return torch.cat(tensors, dim, out=out)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if any([type(a) == ExpandableTensor and isinstance(a._underlying_tensor, FunctionalTensor) for a in args]):
            cm = FunctionalTensorMode()
        else:
            cm = contextlib.nullcontext()
        with cm:
            if func not in _HANDLED_FUNCTIONS or not all(
                issubclass(t, (torch.Tensor, ExpandableTensor)) for t in types
            ):
                args = tuple([a._tensor() if type(a) == ExpandableTensor else a for a in args])
                out = func(*args, **kwargs)
            else:
                out = _HANDLED_FUNCTIONS[func](*args, **kwargs)
        return return_and_correct_aliasing(func, args, kwargs, out)
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        try:
            return expandable_torch_function(func, *args, **kwargs)
        except NotImplementedError:
            pass
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)
    

def expandable_torch_function(func, *args, **kwargs):
    # SDPA is weird, therefore we need to do this before dispatch.
    # Dispatch to the correct implementation here
    if func is torch._C._nn.scaled_dot_product_attention:
        args = [a._tensor() if type(a) == ExpandableTensor else a for a in args]
        return torch._C._nn.scaled_dot_product_attention(*args, **kwargs)

    raise NotImplementedError(func)

class _ExpandableFromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        dim: int,
        preallocate_length: int,
    ) -> "ExpandableTensor":
        return ExpandableTensor(
            input,
            dim=dim,
            preallocate_length=preallocate_length,
        )

    @staticmethod
    def backward(ctx, g0: ExpandableTensor):  # type: ignore[override]
        return g0._underlying_tensor, None, None
    

def create_expandable_tensor(tensor: torch.Tensor, dim, preallocate_length):
    return _ExpandableFromTorchTensor.apply(tensor, dim, preallocate_length)

allow_in_graph(ExpandableTensor)
allow_in_graph(create_expandable_tensor)