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


def _flatten(ids: List[int]):
    result = []
    for id in ids:
        if isinstance(id, list):
            result.extend(_flatten(id))
        else:
            result.append(id)
    return result


class PagedStorage:
    def __init__(
        self,
        storage,
        page_size,
        refcounts,
        static_shape,
    ):
        self.storage = storage
        self.page_size = page_size
        self.refcounts = refcounts
        self.static_shape = static_shape

    @classmethod
    def from_tensor(
        self,
        static_shape,
        dynamic_shape,
        page_size=16,
        initial_storage=None,
        initial_page_ids=None,
        dtype=None,
        device=None,
    ):
        storage = initial_storage
        if storage is None:
            storage = torch.tensor(
                [128, page_size] + static_shape, device=device, dtype=dtype
            )
        elif initial_storage.numel() % (page_size * math.prod(static_shape)):
            last_dynamic_len = dynamic_shape[-1]
            # number of rows left on last dynamic dim to make a whole page
            remaining_rows = page_size - (last_dynamic_len % page_size)
            dyn_copy = dynamic_shape.copy()
            dyn_copy[-1] = remaining_rows
            buffer_shape = dyn_copy + static_shape
            new_storage = torch.zeros(
                buffer_shape, device=storage.device, dtype=storage.dtype
            )
            storage = torch.cat(
                (initial_storage, new_storage), dim=len(dynamic_shape) - 1
            )
            storage = storage.view([-1, page_size] + static_shape)
        else:
            storage = initial_storage.view([-1, page_size] + static_shape)

        pages = int(math.ceil(storage.numel() // page_size))

        refcounts = [0] * pages

        def populate_refcounts(ids):
            if type(ids) == list:
                for id in ids:
                    populate_refcounts(id)
            else:
                refcounts[ids] += 1

        populate_refcounts(initial_page_ids)
        return PagedStorage(storage, page_size, refcounts, static_shape)

    def apply_inplace(self, op_name, args, kwargs, page_ids):
        flattened = _flatten(page_ids)
        for id in flattened:
            op = getattr(self.storage[id], op_name)
            op(*args, **kwargs)

    def apply(self, op, args, kwargs, page_ids):
        storage = PagedStorage(
            self.storage.clone(),
            self.page_size,
            self.refcounts.copy(),
            self.static_shape,
        )
        storage.refcounts = [0 for _ in storage.refcounts]
        for id in _flatten(page_ids):
            storage.refcounts[id] = 1
        storage.apply_inplace(op + "_", args, kwargs, page_ids)
        return storage

    def expand(self, ratio=2, min_pages=128):
        current_pages = self.storage.shape[0]
        page_dim = max(min_pages, current_pages * ratio)
        storage = torch.empty(
            [page_dim, self.page_size] + self.static_shape,
            device=self.storage.device,
            dtype=self.storage.dtype,
        )
        target_slices = tuple(slice(0, size) for size in self.storage.shape)
        storage[target_slices] = self.storage
        self.storage = storage
        added_pages = storage.shape[0] - current_pages
        self.refcounts.extend([0] * added_pages)

    def free_pages(self) -> int:
        """
        How many pages of storage are available for use.
        """
        return sum([1 for x in self.refcounts if x == 0])

    def decrement_refcounts(self, page_ids):
        def decrement(ids):
            if type(ids) == list:
                for id in ids:
                    decrement(id)
            else:
                self.refcounts[ids] -= 1

        decrement(page_ids)


# list from: https://pytorch.org/docs/stable/torch.html
_pointwise_ops = set(
    [
        "abs",
        "absolute",
        "acos",
        "arccos",
        "acosh",
        "arccosh",
        "add",
        "addcdiv",
        "addcmul",
        "angle",
        "asin",
        "arcsin",
        "asinh",
        "arcsinh",
        "atan",
        "arctan",
        "atanh",
        "arctanh",
        "atan2",
        "arctan2",
        "bitwise_not",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_left_shift",
        "bitwise_right_shift",
        "ceil",
        "clamp",
        "clip",
        "conj_physical",
        "copysign",
        "cos",
        "cosh",
        "deg2rad",
        "div",
        "divide",
        "digamma",
        "erf",
        "erfc",
        "erfinv",
        "exp",
        "exp2",
        "expm1",
        "fake_quantize_per_channel_affine",
        "fake_quantize_per_tensor_affine",
        "fix",
        "float_power",
        "floor",
        "floor_divide",
        "fmod",
        "frac",
        "frexp",
        "gradient",
        "imag",
        "ldexp",
        "lerp",
        "lgamma",
        "log",
        "log10",
        "log1p",
        "log2",
        "logaddexp",
        "logaddexp2",
        "logical_and",
        "logical_not",
        "logical_or",
        "logical_xor",
        "logit",
        "hypot",
        "i0",
        "igamma",
        "igammac",
        "mul",
        "multiply",
        "mvlgamma",
        "nan_to_num",
        "neg",
        "negative",
        "nextafter",
        "polygamma",
        "positive",
        "pow",
        "quantized_batch_norm",
        "quantized_max_pool1d",
        "quantized_max_pool2d",
        "rad2deg",
        "real",
        "reciprocal",
        "remainder",
        "round",
        "rsqrt",
        "sigmoid",
        "sign",
        "sgn",
        "signbit",
        "sin",
        "sinc",
        "sinh",
        "softmax",
        "sqrt",
        "square",
        "sub",
        "subtract",
        "tan",
        "tanh",
        "true_divide",
        "trunc",
        "xlogy",
    ]
)


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
        static_dims,
        dynamic_dims,
        page_size,
        current_shape,
        dynamic_shape,
        paged_storage: PagedStorage,
        page_ids,
    ):
        super().__init__()
        self.static_dims = static_dims
        self.dynamic_dims = dynamic_dims
        self.page_size = page_size
        self.current_shape = current_shape
        self.dynamic_shape = dynamic_shape
        self.paged_storage = paged_storage
        self.page_ids = page_ids

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        static_dims: Union[int, List[int]] = -1,
        page_size: int = 16,
    ):
        if isinstance(static_dims, int):
            static_dims = [static_dims]
        else:
            static_dims = static_dims

        # normalize the dims in case of dims like `-1`
        static_dims = [dim % tensor.ndim for dim in static_dims]
        dynamic_dims = [i for i, _ in enumerate(tensor.shape) if i not in static_dims]
        current_shape = list(tensor.shape)
        page_size = page_size
        static_shape = [tensor.shape[x] for x in static_dims]
        dynamic_shape = [tensor.shape[x] for x in dynamic_dims]

        page_ids, _ = _create_page_ids(dynamic_shape, page_size, 0)
        paged_storage = PagedStorage.from_tensor(
            static_shape,
            dynamic_shape,
            page_size,
            tensor,
            page_ids,
            dtype=tensor.dtype,
            device=tensor.dtype,
        )
        return PagedTensor(
            static_dims,
            dynamic_dims,
            page_size,
            current_shape,
            dynamic_shape,
            paged_storage,
            page_ids,
        )

    def __new__(
        cls,
        static_dims,
        dynamic_dims,
        page_size,
        current_shape,
        dynamic_shape,
        paged_storage,
        page_ids,
    ):
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

    def remove(self, idx, dim):
        """
        Modifies self.page_ids, which is an arbitrarily nested list of integers.
        The level of nesting depth is the `dim` (dimension) and `idx` is the index
        to be removed at that level.
        E.g. if self.page_ids = [[1,2,3],[4,5,6]]:
        remove(0, 0) would return [[4,5,6]]
        remove(1,1) would return [[1,3], [4,6]]
        """

        def remove_helper(pages, current_dim):
            if current_dim == dim:
                removed = pages.pop(idx)
                self.paged_storage.decrement_refcounts(removed)
                return pages
            else:
                return [remove_helper(nested, current_dim + 1) for nested in pages]

        self.page_ids = remove_helper(self.page_ids, 0)
        self.current_shape[dim] -= 1
        self.dynamic_shape[dim] -= 1

    # Most of these return a copy of the original
    # tensor, but it handles pointwise ops with look-through
    # into the underlying storage.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        name = func.__qualname__.split(".")[-1]
        inplace = name[-1] == "_"
        if name in _pointwise_ops or (inplace and name[:-1] in _pointwise_ops):
            first_pt = args[0]
            idx = 0
            for i, arg in enumerate(args):
                if isinstance(arg, PagedTensor):
                    first_pt = arg
                    idx = i
                    break
            assert isinstance(first_pt, PagedTensor)
            new_args = args[:idx] + args[idx + 1 :]
            if inplace:
                first_pt.paged_storage.apply_inplace(
                    name, new_args, kwargs, first_pt.page_ids
                )
                return first_pt
            else:
                new_storage = first_pt.paged_storage.apply(
                    name, new_args, kwargs, first_pt.page_ids
                )
                new_pt = PagedTensor(
                    first_pt.static_dims,
                    first_pt.dynamic_dims,
                    first_pt.page_size,
                    first_pt.current_shape,
                    first_pt.dynamic_shape,
                    new_storage,
                    first_pt.page_ids,
                )
                return new_pt

        if func not in _PAGED_HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, PagedTensor)) for t in types
        ):
            args = [a._tensor() if type(a) == PagedTensor else a for a in args]
            return func(*args, **kwargs)
        return _PAGED_HANDLED_FUNCTIONS[func](*args, **kwargs)
