import functools
import itertools
import math
from typing import List, Tuple, TypeAlias, Union

import torch


_EXPANDABLE_HANDLED_FUNCTIONS = {}


def _expandable_implements(torch_function):
    """Register a torch function override"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        _EXPANDABLE_HANDLED_FUNCTIONS[torch_function] = func
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

    def __init__(self, tensor, dim=0, preallocate_length=None):
        super().__init__()
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

    def __new__(cls, tensor, dim=0, preallocate_length=None):
        return super().__new__(cls)

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

    @_expandable_implements(torch.cat)
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
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _EXPANDABLE_HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, ExpandableTensor)) for t in types
        ):
            args = [a._tensor() if type(a) == ExpandableTensor else a for a in args]
            return func(*args, **kwargs)
        return _EXPANDABLE_HANDLED_FUNCTIONS[func](*args, **kwargs)


_PAGED_HANDLED_FUNCTIONS = {}


def _paged_implements(torch_function):
    """Register a torch function override"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        _PAGED_HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


R_LIST: TypeAlias = List[Union[int, "R_LIST"]]
"""
If we treat a tensor's underlying storage as a paged memory space,
This method constructs the appropriate list of "page ids" to index
into storage. The format is a nested list of page_ids where the
outermost list is indexed on the first dynamic shape, etc.

The second part of the tuple returned here is just an implementation
detail used in recursion.
"""


def _create_page_ids(
    dynamic_shapes: List[int], page_size: int, start_page: int
) -> Tuple[R_LIST, int]:
    if len(dynamic_shapes) == 1:
        pages = int(math.ceil(dynamic_shapes[0] / page_size))
        last_page = start_page + pages
        return list(range(start_page, last_page)), last_page
    else:
        result: R_LIST = []
        last_page = start_page
        for _ in range(dynamic_shapes[0]):
            to_add, last_page = _create_page_ids(
                dynamic_shapes[1:], page_size, last_page
            )
            result.append(to_add)
        return result, last_page


class PagedStorage:
    def __init__(
        self,
        static_shape,
        dynamic_shape,
        page_size=16,
        initial_storage=None,
        initial_page_ids=None,
        dtype=None,
        device=None,
    ):
        self.static_shape = static_shape
        self.dynamic_shape = dynamic_shape
        self.page_size = page_size
        self.storage = initial_storage
        if self.storage is None:
            self.storage = torch.tensor(
                [128 * page_size] + self.static_shape, device=device, dtype=dtype
            )
        elif initial_storage.numel() % (page_size * math.prod(self.static_shape)):
            last_dynamic_len = dynamic_shape[-1]
            # number of rows left on last dynamic dim to make a whole page
            remaining_rows = page_size - (last_dynamic_len % page_size)
            dyn_copy = dynamic_shape.copy()
            dyn_copy[-1] = remaining_rows
            buffer_shape = dyn_copy + static_shape
            new_storage = torch.zeros(
                buffer_shape, device=self.storage.device, dtype=self.storage.dtype
            )
            storage = torch.cat(
                (initial_storage, new_storage), dim=len(dynamic_shape) - 1
            )
            self.storage = storage.view([-1, page_size] + self.static_shape)
        else:
            self.storage = initial_storage.view([-1, page_size] + self.static_shape)

        pages = int(math.ceil(self.storage.numel() // page_size))

        self.refcounts = [0] * pages

        def populate_refcounts(ids):
            if type(ids) == list:
                for id in ids:
                    populate_refcounts(id)
            else:
                self.refcounts[ids] += 1

        populate_refcounts(initial_page_ids)

    def expand(self, ratio=2, min_pages=128):
        current_pages = self.storage.numel() // (
            self.page_size * math.prod(self.static_shape)
        )
        page_dim = max(min_pages, current_pages * ratio)
        storage = torch.empty(
            [page_dim, self.page_size] + self.static_shape,
            device=self.storage.device,
            dtype=self.storage.dtype,
        )
        target_slices = tuple(slice(0, size) for size in self.storage.shape)
        storage[target_slices] = self.storage
        self.storage = storage

    def decrement_refcounts(self, page_ids):
        def decrement(ids):
            if type(ids) == list:
                for id in ids:
                    decrement(id)
            else:
                self.refcounts[ids] -= 1

        decrement(page_ids)


class PagedTensor(torch.Tensor):
    """
    A PagedTensor stores tensor data in pages of tensor memory. This allows for
    allocating more space than required, and appending data into an existing
    tensor. It can also allow creating more sophisticated views of the
    underlying storage, which can be useful for kv caching with speculative
    decoding (because we decode many possible continuations of the same cache).

    Args:
    tensor: the tensor to use in paged storage
    static_dims: The dims that won't vary, e.g. an embedding dimension
    page_size: the number of entries of the static portion of the tensor to
        keep in one "page"
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        static_dims: Union[int, List[int]] = -1,
        page_size: int = 16,
    ):
        super().__init__()

        if isinstance(static_dims, int):
            self.static_dims = [static_dims]
        else:
            self.static_dims = static_dims

        # normalize the dims in case of dims like `-1`
        self.static_dims = [dim % tensor.ndim for dim in self.static_dims]
        self.dynamic_dims = [
            i for i, _ in enumerate(tensor.shape) if i not in self.static_dims
        ]
        self.current_shape = list(tensor.shape)
        self.page_size = page_size
        self.static_shape = [tensor.shape[x] for x in self.static_dims]
        self.dynamic_shape = [tensor.shape[x] for x in self.dynamic_dims]
        self.true_page_size = page_size * math.prod(self.static_shape)
        self.min_pages = tensor.numel() // self.true_page_size
        if self.min_pages * self.true_page_size != tensor.numel():
            self.min_pages += 1

        self.page_ids, _ = _create_page_ids(self.dynamic_shape, page_size, 0)
        self.paged_storage = PagedStorage(
            self.static_shape,
            self.dynamic_shape,
            self.page_size,
            tensor,
            self.page_ids,
            dtype=tensor.dtype,
            device=tensor.dtype,
        )

    def __new__(cls, tensor, static_dims=-1, page_size=16):
        return super().__new__(cls)

    def __del__(self):
        self.paged_storage.decrement_refcounts(self.page_ids)

    def size(self, dim=None):
        # https://github.com/pytorch/pytorch/issues/111944
        if dim is None:
            return self.current_shape
        else:
            return self.current_shape[dim]

    def _tensor(self):
        """
        Returns a copy of the tensor, copied from paged storage
        """
        result = torch.empty(
            self.current_shape,
            device=self.paged_storage.storage.device,
            dtype=self.paged_storage.storage.dtype,
        )

        ranges = [range(dim) for dim in self.dynamic_shape[:-1]]
        dynamic_indices = list(itertools.product(*ranges))

        for indices in dynamic_indices:
            page_ids = self.page_ids
            for index in indices:
                page_ids = page_ids[index]
            for i, page_id in enumerate(page_ids):
                complete_index = [slice(None)] * result.ndim
                for dim, index in zip(self.dynamic_dims, indices):
                    complete_index[dim] = index
                last_dynamic = self.dynamic_dims[-1]
                page = self.paged_storage.storage[page_id]

                if i == len(page_ids) - 1:
                    end = self.current_shape[last_dynamic]
                    page_slices = [slice(0, size) for size in page.shape]
                    last_page_size = end - i * self.page_size
                    page_slices[0] = slice(0, last_page_size)
                    page = page[tuple(page_slices)]
                else:
                    end = (i + 1) * self.page_size

                complete_index[last_dynamic] = slice(i * self.page_size, end)
                complete_index = tuple(complete_index)
                result[complete_index] = page

        return result

    def __repr__(self):
        return self._tensor().__repr__()

    # Since this is currently using a copy of the original
    # tensor, these are all handled as out-of-place operations
    # for now.
    # Pointwise ops could all just be pass-through the the appropriate
    # page of the underlying storage, but more complicated ops may need
    # custom kernels.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _PAGED_HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, PagedTensor)) for t in types
        ):
            args = [a._tensor() if type(a) == PagedTensor else a for a in args]
            return func(*args, **kwargs)
        return _PAGED_HANDLED_FUNCTIONS[func](*args, **kwargs)
