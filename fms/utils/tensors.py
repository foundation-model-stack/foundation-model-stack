import functools
from networkx import fast_could_be_isomorphic

from typing import Dict
import torch
from torch._dynamo import allow_in_graph


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
    def __new__(cls, tensor: torch.Tensor, *, dim: int = 0, preallocate_length: int = None):
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            tensor.size(),
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

    def __init__(self, tensor, *, dim=0, preallocate_length=None):
        self._dim = dim
        self._dim_length = tensor.shape[dim]
        self._underlying_tensor = tensor
        if preallocate_length is not None and preallocate_length > self._dim_length:
            sizes = list(tensor.size())
            sizes[dim] = preallocate_length
            self._underlying_tensor = torch.empty(
                size=sizes, dtype=tensor.dtype, device=tensor.device
            )
            self._tensor().copy_(tensor)


    def size(self, dim=None):
        # https://github.com/pytorch/pytorch/issues/111944
        if dim is None:
            return self._tensor().size()
        else:
            return self._tensor().size(dim=dim)

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
                self._tensor(), self._dim, self._underlying_tensor.shape[dim] * 2
            )
            return expanded._append(tensor)

    def _tensor(self):
        """
        Returns a view of the tensor excluding preallocated space
        """
        view = self._underlying_tensor
        sizes = list(view.size())
        sizes[self._dim] = self._dim_length
        view = view.as_strided(size=sizes, stride=view.stride())
        return view

    def __repr__(self):
        return self._tensor().__repr__()

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
        sizes = list(_underlying_tensor.size())
        sizes[_dim] = _dim_length
        return ExpandableTensor(
            _underlying_tensor.as_strided(size=sizes, stride=_underlying_tensor.stride()),
            _dim,
            _underlying_tensor.size(_dim),
        )

    @_implements(torch.cat)
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
        if func not in _HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ExpandableTensor)) for t in types
        ):
            args = [a._tensor() if type(a) == ExpandableTensor else a for a in args]
            return func(*args, **kwargs)
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)
    


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