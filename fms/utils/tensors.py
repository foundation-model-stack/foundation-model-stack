import functools
from typing import Dict, Optional, Union, Tuple, List

import torch
import torch.nn.functional as f
import torch._inductor.ir as ir
import torch._inductor.lowering as lowering
from torch._inductor.virtualized import V
from fms.paged_c import attn_ops, cache_ops  # type: ignore

lib = torch.library.Library("paged_attention", "FRAGMENT")

lib.define(
    "reshape_and_cache_key(Tensor key, Tensor key_cache, Tensor slot_mapping) -> Tensor"
)


# needed for compile
@torch.library.impl(lib, "reshape_and_cache_key", "Meta")
def _reshape_and_cache_key_meta(key, key_cache, slot_mapping):
    return key_cache.contiguous()


@torch.library.impl(lib, "reshape_and_cache_key", "CUDA")
def _reshape_and_cache_key(key, key_cache, slot_mapping):
    key = key.contiguous()
    key_cache = key_cache.contiguous()
    slot_mapping = slot_mapping.contiguous()
    cache_ops.reshape_and_cache_key(key, key_cache, slot_mapping)
    return key_cache

@torch.library.impl(lib, "reshape_and_cache_key", "AutogradNestedTensor")
def _reshape_and_cache_key_autograd(key, key_cache, slot_mapping):
    key = key.contiguous()
    key_cache = key_cache.contiguous()
    slot_mapping = slot_mapping.contiguous()
    reshape = [-1, key_cache.size(3) * key_cache.size(4), key_cache.size(1) * key_cache.size(2)]
    cache_ops.reshape_and_cache_key(key.values().view(*reshape), key_cache, slot_mapping)
    return key_cache


lowering.fallbacks.add(torch.ops.paged_attention.reshape_and_cache_key)


@lowering.register_lowering(
    torch.ops.paged_attention.reshape_and_cache_key, type_promotion_kind=None
)
def _reshape_and_cache_key_lowering(key, key_cache, slot_mapping):
    PagedAttnKernel.create(
        torch.ops.paged_attention.reshape_and_cache_key.default,
        key,
        key_cache,
        slot_mapping,
        mutated_inputs=[key_cache],
    )
    return key_cache

lib.define(
    "reshape_and_cache_value(Tensor value, Tensor value_cache, Tensor slot_mapping) -> Tensor"
)


# needed for compile
@torch.library.impl(lib, "reshape_and_cache_value", "Meta")
def _reshape_and_cache_value_meta(value, value_cache, slot_mapping):
    return value_cache.contiguous()


@torch.library.impl(lib, "reshape_and_cache_value", "CUDA")
def _reshape_and_cache_value(value, value_cache, slot_mapping):
    value = value.contiguous()
    value_cache = value_cache.contiguous()
    slot_mapping = slot_mapping.contiguous()
    cache_ops.reshape_and_cache_value(value, value_cache, slot_mapping)
    return value_cache

@torch.library.impl(lib, "reshape_and_cache_value", "AutogradNestedTensor")
def _reshape_and_cache_value_autograd(value, value_cache, slot_mapping):
    value = value.contiguous()
    value_cache = value_cache.contiguous()
    slot_mapping = slot_mapping.contiguous()
    reshape = [-1, value_cache.size(3), value_cache.size(1) * value_cache.size(2)]
    cache_ops.reshape_and_cache_value(value.values().view(*reshape), value_cache, slot_mapping)
    return value_cache


lowering.fallbacks.add(torch.ops.paged_attention.reshape_and_cache_value)


@lowering.register_lowering(
    torch.ops.paged_attention.reshape_and_cache_value, type_promotion_kind=None
)
def _reshape_and_cache_value_lowering(value, value_cache, slot_mapping):
    PagedAttnKernel.create(
        torch.ops.paged_attention.reshape_and_cache_value.default,
        value,
        value_cache,
        slot_mapping,
        mutated_inputs=[value_cache],
    )
    return value_cache


lib.define(
    "paged_attention_v2(Tensor out, Tensor exp_sums, Tensor max_logits, Tensor tmp_out, Tensor query, Tensor key_cache, Tensor value_cache, int num_kv_heads, float scale, Tensor block_tables, Tensor context_lens, int block_size, SymInt max_context_len, Tensor? alibi_slopes) -> Tensor"
)


@torch.library.impl(lib, "paged_attention_v2", "Meta")
def _paged_attention_v2_meta(
    out,
    exp_sums,
    max_logits,
    tmp_out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    return out.contiguous()


@torch.library.impl(lib, "paged_attention_v2", "CUDA")
def _paged_attention_v2(
    out,
    exp_sums,
    max_logits,
    tmp_out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    out = out.contiguous()
    exp_sums = exp_sums.contiguous()
    max_logits = max_logits.contiguous()
    tmp_out = tmp_out.contiguous()
    query = query.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    block_tables = block_tables.contiguous()
    context_lens = context_lens.contiguous()

    attn_ops.paged_attention_v2(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
    )
    return out


lowering.fallbacks.add(torch.ops.paged_attention.paged_attention_v2)


@lowering.register_lowering(
    torch.ops.paged_attention.paged_attention_v2, type_promotion_kind=None
)
def _paged_attention_v2_lowering(
    out,
    exp_sums,
    max_logits,
    tmp_out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    PagedAttnKernel.create(
        torch.ops.paged_attention.paged_attention_v2.default,
        out,
        exp_sums,
        max_logits,
        tmp_out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        mutated_inputs=[out],
    )
    return out


lib.define(
    "paged_attention_v1(Tensor out, Tensor query, Tensor key_cache, Tensor value_cache, int num_kv_heads, float scale, Tensor block_tables, Tensor context_lens, int block_size, SymInt max_context_len, Tensor? alibi_slopes) -> Tensor"
)


@torch.library.impl(lib, "paged_attention_v1", "Meta")
def _paged_attention_v1_meta(
    out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    return out.contiguous()


@torch.library.impl(lib, "paged_attention_v1", "CUDA")
def _paged_attention_v1(
    out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    attn_ops.paged_attention_v1(
        out,
        query,
        key_cache.blob,
        value_cache.blob,
        key_cache.kv_heads,
        scale,
        key_cache.block_mapping,
        key_cache.context_lengths,
        key_cache.block_size,
        max_context_len,
        None,
    )
    return out

lib.define(
    "paged_attention_v1_minimal(Tensor query, Tensor paged_key, Tensor paged_value, float scale) -> Tensor"
)
@torch.library.impl(lib, "paged_attention_v1_minimal", "AutogradNestedTensor")
def _paged_attention_v1_minimal_autograd(
    query,
    paged_key,
    paged_value,
    scale,
):
    out = torch.empty_like(query)

    attn_ops.paged_attention_v1(
        out,
        query,
        paged_key.blob,
        paged_value.blob,
        paged_key.kv_heads,
        scale,
        paged_key.block_mapping,
        paged_key.context_lengths,
        paged_key.block_size,
        torch.max(paged_key.context_lengths).item(),
        None,
    )
    return out


lowering.fallbacks.add(torch.ops.paged_attention.paged_attention_v1)


@lowering.register_lowering(
    torch.ops.paged_attention.paged_attention_v1, type_promotion_kind=None
)
def _paged_attention_v1_lowering(
    out,
    query,
    key_cache,
    value_cache,
    num_kv_heads,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes=None,
):
    PagedAttnKernel.create(
        torch.ops.paged_attention.paged_attention_v1.default,
        out,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes,
        mutated_inputs=[out],
    )
    return out


# Available starting PT 2.2
class NoneLayout(ir.IRNode):
    def __init__(self, device):
        self.device = device
        self.size = [0]
        self.stride = [0]

    def storage_size(self):
        return 0

    def as_fixed(self):
        return self


# Available starting PT 2.2
class MutationOutput(ir.ExternKernel):
    def get_mutation_names(self):
        return [self.inputs[0].get_name()]

    def __init__(self, layout, input, parent):
        super().__init__(None, layout, [input, parent], ())
        self.name = V.graph.register_buffer(self)

    def should_allocate(self):
        return False

    def is_no_op(self):
        return True

    def has_side_effects(self):
        return True

    def get_alias_names(self):
        return [self.inputs[0].get_name()]


class PagedAttnKernel(ir.FallbackKernel):
    def should_allocate(self):
        return False

    def has_side_effects(self):
        return True

    @classmethod
    def create(cls, kernel, *args, mutated_inputs=[], **kwargs) -> None:
        with V.graph.fake_mode:
            (
                example_output,
                tensor_args,
                non_tensor_args,
                unflatten_args,
                schema,
            ) = cls.process_kernel(kernel, *args, **kwargs)
        for tensor_arg in tensor_args:
            tensor_arg.realize()

        packed = cls(
            NoneLayout(tensor_args[0].get_device()),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            schema=schema,
        )
        # Mark inplace inputs as mutated
        for kernel_input in mutated_inputs:
            V.graph.mark_buffer_mutated(kernel_input.get_name())
            MutationOutput(kernel_input.layout, kernel_input, packed)

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
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_FUNCTIONS or not all(
                issubclass(t, (torch.Tensor, ExpandableTensor)) for t in types
        ):
            args = [a._tensor() if type(a) == ExpandableTensor else a for a in args]
            return func(*args, **kwargs)
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __tensor_flatten__(self):
        ctx = {
            "dim": self._dim,
        }

        inner_tensors = ["_underlying_tensor"]
        return inner_tensors, ctx

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride):
        underlying_tensor = inner_tensors["_underlying_tensor"]
        dim = meta["dim"]

        return ExpandableTensor(underlying_tensor, dim=dim)

class PagedTensor(torch.Tensor):

    def __init__(self, kv_heads: int, head_size: int, num_blocks: int, is_key: bool, block_size: int = 16, dtype: torch.dtype = torch.float32, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        device = "cuda"
        # TODO: is_key name change
        # TODO: flexible sizing

        element_size = torch.tensor([], dtype=dtype).element_size()
        x = block_size // element_size
        if is_key:
            block_shape = (
                kv_heads,
                head_size // x,
                block_size,
                x
            )
        else:
            block_shape = (
                kv_heads,
                head_size,
                block_size
            )
        self.kv_heads = kv_heads
        self.head_size = head_size
        self.num_blocks = num_blocks
        self.is_key = is_key
        self.block_size = block_size
        self._device = device
        self.blob = torch.empty(
            size=(num_blocks, *block_shape),
            dtype=dtype,
            device=device
        )

        self.ref_counts = torch.zeros(num_blocks, dtype=torch.int32, device=device)
        self.context_lengths = torch.empty(0, dtype=torch.int32, device=device)
        self.block_mapping = torch.empty(0, 0, dtype=torch.int32, device=device)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in _HANDLED_FUNCTIONS or not all(
                issubclass(t, (torch.Tensor, PagedTensor)) for t in types
        ):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    def _allocate_blocks(self, num_blocks) -> torch.Tensor:
        # TODO: keep num_blocks on gpu
        first_available_blocks = (self.ref_counts == 0).nonzero()[0:int(num_blocks.item())] + 1
        self.ref_counts[first_available_blocks - 1] = 1
        return first_available_blocks.squeeze(1).int()

    def __repr__(self):
        return f"PagedTensor(context_lengths={self.context_lengths}, block_mapping={self.block_mapping.tolist()})"

    @_implements(torch.cat)
    def cat(tensors, *, dim=0, out=None):

        # if we are calling with dim=0, a new sequence is being added to the batch
        if dim == 0:


            # if the second tensor is not a paged tensor, we are not doing a merge in which case we need to just add
            # sequences to the paged tensor
            if not isinstance(tensors[1], PagedTensor):
                l_tensors = tensors[1]

                # only allow empty for now
                assert l_tensors.size(1) == 0

                # TODO: will we even support tensor[0] being non paged_tensor, is that possible???
                paged_tensor: PagedTensor = tensors[0]

                # TODO: This needs to be right padded
                paged_tensor.block_mapping = torch.cat((paged_tensor.block_mapping, torch.empty(l_tensors.size(0), 0, device=paged_tensor._device)), dim=0)

                paged_tensor.context_lengths = torch.cat((paged_tensor.context_lengths,torch.zeros(l_tensors.size(0), device=paged_tensor._device)))
            else:
                pass
        # decode step
        elif dim == 1:
            l_tensors = tensors[1].unbind()  # TODO: assume nested

            # TODO: will we even support tensor[0] being non paged_tensor, is that possible???
            paged_tensor: PagedTensor = tensors[0]

            # for now it's a list
            # TODO: this is a list just to get things working
            context_lengths = torch.tensor([l.size(0) for l in l_tensors], dtype=torch.int32, device=paged_tensor._device)

            # batch should match
            assert len(l_tensors) == paged_tensor.block_mapping.size(0)

            prev_context_lengths = paged_tensor.context_lengths
            total_context_lengths = prev_context_lengths + context_lengths

            # get the total old number of blocks in the block mapping (not including pads)
            prev_num_blocks = torch.ceil(prev_context_lengths / paged_tensor.block_size)

            # get the total new number of blocks in the block mapping (not including pads)
            total_num_blocks = torch.ceil(total_context_lengths / paged_tensor.block_size)

            # pre-pad block_mapping to max
            pad = (0, (torch.max(total_num_blocks) - paged_tensor.block_mapping.size(1)).int().tolist())
            paged_tensor.block_mapping = f.pad(input=paged_tensor.block_mapping, pad=pad, mode='constant', value=0).int()

            n_blocks_to_add = (total_num_blocks - prev_num_blocks).int()
            blocks_to_add = paged_tensor._allocate_blocks(torch.sum(n_blocks_to_add))

            # CALCULATE ROW INDICES
            # At what new-block indices do we enter a new row?
            rowbreak_thresh = n_blocks_to_add.cumsum(0)
            # A pseudo-arange representing new block indices (stays on gpu)
            newblock_inds = torch.ones_like(blocks_to_add).cumsum(0).unsqueeze(0).sub(1)
            # What row break thresholds does each index position qualify for
            below_thresh = newblock_inds < rowbreak_thresh.unsqueeze(1)

            # The first qualifying break represents the assigned sequence
            row_inds = below_thresh.int().argmax(0)

            # CALCULATE COLUMN INDICES
            col_offsets = paged_tensor.block_mapping.count_nonzero(1)

            # Does each new value belong to the same row as the prior one?
            row_match = row_inds.roll(1) == row_inds
            # Count the total number of row agreements
            col_increments = row_match.int().cumsum(0)
            # Subtract the number of agreements from prior rows to get within-row increments
            col_increments -= f.pad(n_blocks_to_add.sub(1).clamp(min=0).cumsum(0), (1, 0))[:-1][row_inds]
            # Add the column offset for each row
            col_inds = col_increments + col_offsets[row_inds]

            paged_tensor.block_mapping[row_inds, col_inds] = blocks_to_add
            paged_tensor.context_lengths = total_context_lengths.int()

            block_slots = paged_tensor.block_mapping.repeat_interleave(paged_tensor.block_size, dim=1)
            roll_inds = torch.ones_like(block_slots).cumsum(1).sub(1)
            roll_inds = roll_inds.add(paged_tensor.context_lengths.unsqueeze(1)) % roll_inds.size(1)
            slot_mask = roll_inds.sign().cumprod(1)

            block_offset = (roll_inds % paged_tensor.block_size) * (1 - slot_mask)
            block_base = block_slots.gather(1, roll_inds) * (1 - slot_mask)
            slot_map = block_base * paged_tensor.block_size + block_offset + slot_mask.neg()

            if paged_tensor.is_key:
                # call reshape_and_cache_key
                data_layer = torch.ops.paged_attention.reshape_and_cache_key(
                    tensors[1],
                    paged_tensor.blob,
                    slot_map[slot_map != -1],
                )
            else:
                # call reshape_and_cache_value
                data_layer = torch.ops.paged_attention.reshape_and_cache_value(
                    tensors[1],
                    paged_tensor.blob,
                    slot_map[slot_map != -1],
                )

        else:
            raise ValueError("PagedTensor only supports dimensions 0 and 1 for dim parameter")

        return paged_tensor

if __name__ == "__main__":
    pt = PagedTensor(8, 64, 1000, False)

    # warm up sequences
    pt = torch.cat((pt, torch.empty(4, 0, device="cuda")), dim=0)
    print(pt.block_mapping)
    print(pt.context_lengths)

    l = [
        torch.randn(100,64,1024, device="cuda"),
        torch.randn(40,64,1024, device="cuda"),
        torch.randn(8,64,1024, device="cuda"),
        torch.randn(24,64,1024, device="cuda"),
    ]
    l = torch.nested.nested_tensor(l, device="cuda")

    pt2 = torch.cat((pt, l), dim=1)
    print(f"block_mapping pt2: {pt.block_mapping}")
    print(pt.context_lengths)

    l2 = [
        torch.randn(1, 64, 1024, device="cuda"),
        torch.randn(1, 64, 1024, device="cuda"),
        torch.randn(1, 64, 1024, device="cuda"),
        torch.randn(1, 64, 1024, device="cuda"),
    ]
    l2 = torch.nested.nested_tensor(l2, device="cuda")
    pt3 = torch.cat((pt2, l2), dim=1)
    print(f"block_mapping pt3: {pt.block_mapping}")
    print(pt.context_lengths)

    
    # block_positions = (torch.ones_like(block_mapping_repeated).cumsum(dim=1) - 1)# % pt2.block_size
    # print(block_positions)
    #
    # block_slots = bmap.repeat_interleave(16, dim=1)
    # roll_inds = torch.ones_like(block_slots).cumsum(1).sub(1)
    # roll_inds = roll_inds.add(clen.unsqueeze(1)) % roll_inds.size(1)
    # slot_mask = roll_inds.sign().cumprod(1)
    #
    # slot_map = block_base * 16 + block_offset + slot_mask.neg()

    # print(pt2.block_mapping.repeat_interleave(pt2.block_size, dim=1))
    # print(block_positions)

    # def get_slot_mapping(self, position: Optional[int] = None) -> List[int]:
    #     slot_mapping = []
    #     start = position if position else 0
    #     for position_i in range(start, self.get_sequence_length()):
    #         block_number = self.get_cache_block(position_i).block_number
    #         block_offset = position_i % self.block_size
    #         slot = block_number * self.block_size + block_offset
    #         slot_mapping.append(slot)
    #     return slot_mapping

    # pt3 = torch.cat((pt2, torch.rand(4, 200)), dim=1)
    # print(f"block_mapping pt3: {pt3.block_mapping}")
    # print(pt3.context_lengths)

    #


    #
    # position_ids = torch.tensor([
    #     [0 for _ in range(5)] + [i for i in range(15)],
    #     [0 for _ in range(8)] + [i for i in range(12)],
    #     [0 for _ in range(10)] + [i for i in range(10)],
    #     [i for i in range(20)],
    #     [0 for _ in range(5)] + [i for i in range(15)],
    #     [0 for _ in range(5)] + [i for i in range(15)],
    #     [0 for _ in range(5)] + [i for i in range(15)],
    #     [i for i in range(20)],
    # ], dtype=torch.long, device="cpu")
    # pt = torch.cat((pt, key), dim=1, position_ids=position_ids)
    # print(pt.block_mapping)
    # print(pt.context_lengths)
    # print(pt.ref_counts)
    #
    # # warmup
    # # warm up sequences
    # pt = torch.cat((pt, torch.empty(8, 0)), dim=0)
    #
    # # MHA
    #
    # # key_computed is a nested tensor that is jagged
    # # if not a nested tensor, then assume largest length
    # # calls paged attention store kernel with key_computed
    # key_cache = torch.cat((key_cache, key_computed), dim=1)